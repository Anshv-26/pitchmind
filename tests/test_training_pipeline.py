"""Tests for the Stage 1 training/evaluation layer.

This module skips entirely (not an error) if scikit-learn/xgboost/catboost
are not yet installed, so it never blocks collection of
tests/test_feature_engineering.py while dependencies are being set up.

Strategy: a small, fast synthetic Premier League history using REAL season
labels (2015_16 through 2024_25) drives most tests, so the real
`INTENDED_TRAINING_CUTOFF` pairing table applies to it directly without
needing the actual 4,180-row dataset. A handful of tests also exercise the
real, already-built fold artifacts on disk as integration tests, skipped if
those files are not present.

Leakage canary: `test_preprocessing_is_fit_only_on_training_rows_leakage_canary`
mutates validation features dramatically and proves the fitted
imputer/scaler statistics do not move.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")
xgboost = pytest.importorskip("xgboost")
catboost = pytest.importorskip("catboost")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.app.ml.baselines as baselines  # noqa: E402
import backend.app.ml.datasets as datasets  # noqa: E402
import backend.app.ml.evaluation as evaluation  # noqa: E402
import backend.app.ml.training as training  # noqa: E402
from backend.app.ml.datasets import DEVELOPMENT_FOLDS, load_fold  # noqa: E402
from backend.app.ml.feature_engineering import (  # noqa: E402
    DEFAULT_EFFICIENCY_WINDOW,
    DEFAULT_EWMA_HALFLIFE,
    DEFAULT_MIN_PERIODS,
    DEFAULT_REST_DAYS_CAP,
    FEATURE_COLUMNS,
    LINEAR_SAFE_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RESULT_COLUMN,
    SEALED_SEASON,
    TARGET_COLUMN,
    assert_artifact_valid_for,
    build_causal_elo_schedule,
    build_features,
    build_provenance,
    canonical_elo_params,
    sidecar_path_for,
)

REAL_FOLD1_PATH = REPO_ROOT / "data" / "processed" / "features_causal_through_2021_22.parquet"
requires_real_artifacts = pytest.mark.skipif(
    not REAL_FOLD1_PATH.exists(), reason="real fold artifacts not built"
)


# --------------------------------------------------------------------------
# Synthetic Premier League history (real season labels, small scale)
# --------------------------------------------------------------------------
def _build_synthetic_pl_league(season_teams: dict[str, list[str]]) -> pd.DataFrame:
    """A tiny synthetic match history using REAL season labels, so the real
    INTENDED_TRAINING_CUTOFF pairing table applies to it directly.

    Takes a per-season roster (rather than one fixed team list) so the
    fixture can include a genuine newcomer team-season - without at least
    one, `estimate_elo_params` correctly refuses to estimate the
    promoted/returning prior, since there is nothing to estimate it from.
    """
    records = []
    for season_index, (season, teams) in enumerate(season_teams.items()):
        start = pd.Timestamp(f"{season.split('_')[0]}-08-10")
        day = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                if (day + season_index) % 3 == 0:
                    fthg, ftag = 1, 1
                elif day % 2 == 0:
                    fthg, ftag = 2, 0
                else:
                    fthg, ftag = 0, 2
                ftr = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
                records.append(
                    {
                        "Season": season,
                        "Date": start + pd.Timedelta(days=7 * day),
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": fthg,
                        "FTAG": ftag,
                        "FTR": ftr,
                        "HS": 10,
                        "AS": 10,
                        "HST": 5,
                        "AST": 5,
                    }
                )
                day += 1
    return pd.DataFrame(records)


SYNTHETIC_SEASONS = [
    "2015_16", "2016_17", "2017_18", "2018_19", "2019_20",
    "2020_21", "2021_22", "2022_23", "2023_24", "2024_25",
]
# Five ever-present teams (A-E) plus one roster change after the opening
# season: F (2015/16 only) is replaced by G from 2016/17 onward. This gives
# estimate_elo_params exactly one genuine newcomer team-season (G in
# 2016/17) to compute the promoted/returning prior from, while keeping five
# continuing teams for the season-to-season shrink regression. Total match
# count is unchanged (30 fixtures/season, 6 teams throughout).
SYNTHETIC_SEASON_TEAMS: dict[str, list[str]] = {
    season: (["A", "B", "C", "D", "E", "F"] if season == SYNTHETIC_SEASONS[0] else ["A", "B", "C", "D", "E", "G"])
    for season in SYNTHETIC_SEASONS
}


@pytest.fixture(scope="module")
def synthetic_matches() -> pd.DataFrame:
    return _build_synthetic_pl_league(SYNTHETIC_SEASON_TEAMS)


@pytest.fixture(scope="module")
def synthetic_schedule(synthetic_matches):
    return build_causal_elo_schedule(synthetic_matches)


@pytest.fixture(scope="module")
def synthetic_features(synthetic_matches, synthetic_schedule) -> pd.DataFrame:
    return build_features(synthetic_matches, synthetic_schedule)


def _season_order(frame: pd.DataFrame) -> list[str]:
    return list(frame.groupby("Season")["Date"].min().sort_values().index)


def _write_fold_artifact(
    directory: Path, features: pd.DataFrame, schedule, *, cutoff: str, eval_season: str
) -> Path:
    order = _season_order(features)
    keep = set(order[: order.index(eval_season) + 1])
    capped = features[features["Season"].isin(keep)].reset_index(drop=True)
    path = directory / f"features_causal_through_{cutoff}.parquet"
    capped.to_parquet(path, index=False)
    provenance = build_provenance(
        elo_params=schedule,
        ewma_halflife=DEFAULT_EWMA_HALFLIFE,
        min_periods=DEFAULT_MIN_PERIODS,
        efficiency_window=DEFAULT_EFFICIENCY_WINDOW,
        rest_days_cap=DEFAULT_REST_DAYS_CAP,
        source_path="synthetic",
        source_sha256="deadbeef",
        n_rows=len(capped),
        purpose="test fold artifact",
        valid_for_evaluation=True,
        intended_evaluation_seasons=[eval_season],
        artifact_row_cap_season=eval_season,
        notes="",
    )
    sidecar_path_for(path).write_text(json.dumps(provenance, indent=2))
    return path


def _write_canonical_artifact(directory: Path, matches: pd.DataFrame) -> Path:
    features = build_features(matches, canonical_elo_params())
    path = directory / "features.parquet"
    features.to_parquet(path, index=False)
    provenance = build_provenance(
        elo_params=canonical_elo_params(),
        ewma_halflife=DEFAULT_EWMA_HALFLIFE,
        min_periods=DEFAULT_MIN_PERIODS,
        efficiency_window=DEFAULT_EFFICIENCY_WINDOW,
        rest_days_cap=DEFAULT_REST_DAYS_CAP,
        source_path="synthetic",
        source_sha256="deadbeef",
        n_rows=len(features),
        purpose="canonical",
        valid_for_evaluation=False,
        notes="not valid for evaluation",
    )
    sidecar_path_for(path).write_text(json.dumps(provenance, indent=2))
    return path


@pytest.fixture()
def synthetic_processed_dir(tmp_path, synthetic_matches, synthetic_schedule, synthetic_features, monkeypatch):
    """A tmp `data/processed`-equivalent directory holding all 3 development
    fold artifacts plus the canonical artifact, so `datasets.load_fold` can
    be exercised end-to-end without touching the real repo's data."""
    for cutoff, eval_season in [("2021_22", "2022_23"), ("2022_23", "2023_24"), ("2023_24", "2024_25")]:
        _write_fold_artifact(tmp_path, synthetic_features, synthetic_schedule, cutoff=cutoff, eval_season=eval_season)
    _write_canonical_artifact(tmp_path, synthetic_matches)
    monkeypatch.setattr(datasets, "PROCESSED_DIR", tmp_path)
    return tmp_path


# ==========================================================================
# 1-2. Artifact safety: canonical rejected, wrong fold rejected, correct
# fold accepted (items 1-3)
# ==========================================================================
def test_canonical_artifact_rejected_for_evaluation(synthetic_processed_dir):
    canonical_path = synthetic_processed_dir / "features.parquet"
    with pytest.raises(ValueError, match="not valid for model evaluation"):
        assert_artifact_valid_for(canonical_path, ["2022_23"])


def test_load_fold_has_no_path_to_the_canonical_artifact():
    import inspect

    assert list(inspect.signature(load_fold).parameters) == ["fold_number"]
    for fold_number in DEVELOPMENT_FOLDS:
        path = datasets.artifact_path_for_fold(fold_number)
        assert path.name != "features.parquet"
        assert "causal_through" in path.name


def test_wrong_fold_artifact_rejected_when_validation_season_missing(
    synthetic_processed_dir, synthetic_features, synthetic_schedule
):
    # Simulate a mis-built fold-1 artifact that never reached its own
    # validation season (e.g. built with the wrong --causal-through cutoff).
    _write_fold_artifact(
        synthetic_processed_dir, synthetic_features, synthetic_schedule,
        cutoff="2021_22", eval_season="2021_22",
    )
    with pytest.raises(ValueError, match="does not contain evaluation season"):
        load_fold(1)


def test_correct_fold_artifact_accepted(synthetic_processed_dir):
    fold = load_fold(1)
    assert fold.fold == 1
    assert fold.validation_season == "2022_23"
    assert fold.train_seasons[-1] == "2021_22"
    assert len(fold.X_train) > 0
    assert len(fold.X_val) > 0


@requires_real_artifacts
def test_real_development_folds_load_and_contain_no_sealed_rows():
    for fold_number in DEVELOPMENT_FOLDS:
        fold = load_fold(fold_number)
        seasons_present = set(fold.metadata_train["Season"]) | set(fold.metadata_val["Season"])
        assert SEALED_SEASON not in seasons_present


# ==========================================================================
# 4-5. No sealed rows; train strictly precedes validation (items 4-5)
# ==========================================================================
def test_development_artifacts_contain_no_sealed_rows(synthetic_processed_dir):
    for fold_number in DEVELOPMENT_FOLDS:
        fold = load_fold(fold_number)
        seasons_present = set(fold.metadata_train["Season"]) | set(fold.metadata_val["Season"])
        assert SEALED_SEASON not in seasons_present
        assert fold.validation_season not in set(fold.metadata_train["Season"])


def test_train_rows_strictly_precede_validation_rows(synthetic_processed_dir):
    for fold_number in DEVELOPMENT_FOLDS:
        fold = load_fold(fold_number)
        assert fold.metadata_train["Date"].max() < fold.metadata_val["Date"].min()


def test_exactly_one_validation_season_per_fold(synthetic_processed_dir):
    for fold_number in DEVELOPMENT_FOLDS:
        fold = load_fold(fold_number)
        assert set(fold.metadata_val["Season"].unique()) == {fold.validation_season}


# ==========================================================================
# 6-8. Metadata / FTR / target never enter X
# ==========================================================================
def test_metadata_ftr_and_target_never_enter_X(synthetic_processed_dir):
    fold = load_fold(1)
    for col in METADATA_COLUMNS:
        assert col not in fold.X_train.columns
        assert col not in fold.X_val.columns
    assert RESULT_COLUMN not in fold.X_train.columns
    assert TARGET_COLUMN not in fold.X_train.columns
    assert RESULT_COLUMN not in fold.X_val.columns
    assert TARGET_COLUMN not in fold.X_val.columns
    assert list(fold.X_train.columns) == FEATURE_COLUMNS
    assert list(fold.X_val.columns) == FEATURE_COLUMNS


# ==========================================================================
# 9-10. Feature-subset contracts per model family
# ==========================================================================
def test_tree_models_receive_all_35_features():
    for name in ["random_forest", "xgboost", "catboost"]:
        spec = training.MODEL_SPECS[name]
        assert len(spec.feature_columns) == 35
        assert set(spec.feature_columns) == set(FEATURE_COLUMNS)


def test_logreg_receives_25_base_features_before_indicator_columns(synthetic_processed_dir):
    spec = training.MODEL_SPECS["logreg"]
    assert len(spec.feature_columns) == 25
    assert set(spec.feature_columns) == set(LINEAR_SAFE_FEATURE_COLUMNS)

    fold = load_fold(1)
    pipeline = spec.build({"C": 1.0})
    pipeline.fit(fold.X_train[list(spec.feature_columns)], fold.y_train)

    assert pipeline.named_steps["imputer"].n_features_in_ == 25
    # The imputer may ADD indicator columns downstream (25 + however many
    # columns had missing values); the model must receive at least the base 25.
    assert pipeline.named_steps["scaler"].n_features_in_ >= 25


# ==========================================================================
# 11-13. Train-only preprocessing + leakage canary
# ==========================================================================
def test_preprocessing_is_fit_only_on_training_rows_leakage_canary(synthetic_processed_dir):
    fold = load_fold(1)
    spec = training.MODEL_SPECS["logreg"]
    columns = list(spec.feature_columns)

    baseline_pipeline = spec.build({"C": 1.0})
    baseline_pipeline.fit(fold.X_train[columns], fold.y_train)
    baseline_medians = baseline_pipeline.named_steps["imputer"].statistics_.copy()
    baseline_mean = baseline_pipeline.named_steps["scaler"].mean_.copy()
    baseline_scale = baseline_pipeline.named_steps["scaler"].scale_.copy()

    # Dramatically mutate validation features - this must never be seen by fit().
    mutated_val = fold.X_val.copy()
    mutated_val.loc[:, columns] = mutated_val.loc[:, columns] * 1000.0 + 1_000_000.0

    mutated_pipeline = spec.build({"C": 1.0})
    mutated_pipeline.fit(fold.X_train[columns], fold.y_train)  # only X_train, never mutated_val

    np.testing.assert_array_equal(baseline_medians, mutated_pipeline.named_steps["imputer"].statistics_)
    np.testing.assert_array_equal(baseline_mean, mutated_pipeline.named_steps["scaler"].mean_)
    np.testing.assert_array_equal(baseline_scale, mutated_pipeline.named_steps["scaler"].scale_)

    # And the mutation DID reach predict_proba (proving it wasn't just inert),
    # so the equality above is a real leakage check, not a vacuous one.
    proba_original = baseline_pipeline.predict_proba(fold.X_val[columns])
    proba_mutated = mutated_pipeline.predict_proba(mutated_val[columns])
    assert not np.allclose(proba_original, proba_mutated)


def test_random_forest_imputer_also_fits_train_only(synthetic_processed_dir):
    fold = load_fold(1)
    spec = training.MODEL_SPECS["random_forest"]
    columns = list(spec.feature_columns)
    params = spec.param_grid[0]

    a = spec.build(params)
    a.fit(fold.X_train[columns], fold.y_train)
    b = spec.build(params)
    b.fit(fold.X_train[columns], fold.y_train)

    np.testing.assert_array_equal(
        a.named_steps["imputer"].statistics_, b.named_steps["imputer"].statistics_
    )


# ==========================================================================
# 14-17. Probability contract
# ==========================================================================
def test_probability_contract_for_every_model(synthetic_processed_dir):
    fold = load_fold(1)
    for spec in training.MODEL_SPECS.values():
        params = spec.param_grid[0]
        proba = training.fit_predict(spec, params, fold)
        assert proba.shape == (len(fold.X_val), 3)
        assert np.isfinite(proba).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert evaluation.validate_probabilities(proba) == []


def test_probability_contract_for_every_baseline(synthetic_processed_dir):
    fold = load_fold(1)
    predictors = {
        "class_frequency": baselines.class_frequency_baseline(fold.y_train, len(fold.y_val)),
        "elo_only": baselines.elo_only_baseline(fold.X_train, fold.y_train, fold.X_val),
        "strength_trio": baselines.strength_trio_baseline(fold.X_train, fold.y_train, fold.X_val),
    }
    for name, proba in predictors.items():
        assert proba.shape == (len(fold.X_val), 3), name
        assert np.isfinite(proba).all(), name
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), name


def test_class_order_is_always_H0_D1_A2(synthetic_processed_dir):
    fold = load_fold(1)
    spec = training.MODEL_SPECS["logreg"]
    pipeline = spec.build({"C": 1.0})
    pipeline.fit(fold.X_train[list(spec.feature_columns)], fold.y_train)
    assert list(pipeline.named_steps["model"].classes_) == [0, 1, 2]


def test_align_proba_handles_missing_classes():
    proba = np.array([[0.6, 0.4], [0.3, 0.7]])  # only classes 0 and 2 were present
    aligned = evaluation.align_proba_to_expected_classes(proba, classes=[0, 2])
    assert aligned.shape == (2, 3)
    np.testing.assert_allclose(aligned[:, 1], 0.0)  # draw column zero-filled
    np.testing.assert_allclose(aligned[:, 0], [0.6, 0.3])
    np.testing.assert_allclose(aligned[:, 2], [0.4, 0.7])


def test_align_proba_renormalizes_float32_precision_noise():
    """Regression test for the XGBoost 'y_prob values do not sum to one'
    warning: a float32 softmax naturally leaves row sums ~1e-7 off 1.0;
    upcasting to float64 without renormalizing preserves that noise as
    float64 values, which is enough to trip sklearn's dtype-derived
    log_loss tolerance even though nothing is actually wrong with the
    probabilities."""
    raw = np.array(
        [[0.42911652, 0.31038135, 0.26050222],
         [0.48823035, 0.26298022, 0.24878936],
         [0.24351029, 0.28034803, 0.47614175]],
        dtype=np.float32,
    )
    # Perturb by ~float32 eps so row sums deviate the way XGBoost's native
    # output does (~1e-7), without relying on an installed model to reproduce it.
    raw[:, 0] += np.float32(1.1920929e-07)
    row_sums_before = raw.astype(np.float64).sum(axis=1)
    assert np.max(np.abs(row_sums_before - 1.0)) > 1e-8  # the noise is real and present

    aligned = evaluation.align_proba_to_expected_classes(raw, classes=[0, 1, 2])

    assert aligned.dtype == np.float64
    row_sums_after = aligned.sum(axis=1)
    # Strict: within float64 machine precision, not just within the loose
    # 1e-6 validate_probabilities tolerance.
    np.testing.assert_allclose(row_sums_after, 1.0, atol=1e-12, rtol=0)
    assert evaluation.validate_probabilities(aligned) == []

    # Class order [H, D, A] = [0, 1, 2] is unchanged: column i still
    # corresponds to the same class, just rescaled by a per-row constant.
    ratios = aligned / raw.astype(np.float64)
    for row in range(raw.shape[0]):
        np.testing.assert_allclose(ratios[row], ratios[row, 0], rtol=1e-6)

    # Argmax predictions are unchanged - renormalizing by a single positive
    # per-row scalar cannot reorder a row's own values.
    np.testing.assert_array_equal(raw.argmax(axis=1), aligned.argmax(axis=1))


def test_align_proba_rejects_a_non_positive_row_sum():
    zero_row = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match="non-positive sum"):
        evaluation.align_proba_to_expected_classes(zero_row, classes=[0, 1, 2])


def test_align_proba_rejects_a_non_finite_row_sum():
    nan_row = np.array([[0.5, np.nan, 0.2], [0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match="non-finite sum"):
        evaluation.align_proba_to_expected_classes(nan_row, classes=[0, 1, 2])


def test_validate_probabilities_catches_bad_shape():
    assert evaluation.validate_probabilities(np.zeros((5, 2))) != []


def test_validate_probabilities_catches_bad_sum():
    bad = np.tile([0.5, 0.5, 0.5], (3, 1))
    assert evaluation.validate_probabilities(bad) != []


def test_validate_probabilities_catches_non_finite():
    bad = np.array([[0.5, 0.3, np.nan]])
    assert evaluation.validate_probabilities(bad) != []


def test_validate_probabilities_accepts_a_sound_array():
    good = np.tile([0.5, 0.3, 0.2], (4, 1))
    assert evaluation.validate_probabilities(good) == []


def test_multiclass_brier_score_hand_computed():
    y_true = np.array([0, 1])
    perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert evaluation.multiclass_brier_score(y_true, perfect) == pytest.approx(0.0)

    fully_wrong = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    # squared error per row: (0-1)^2+(0-0)^2+(1-0)^2 = 2; mean over 2 rows = 2.0
    assert evaluation.multiclass_brier_score(y_true, fully_wrong) == pytest.approx(2.0)


# ==========================================================================
# 18-19. Class-frequency baseline train-only, validation-blind
# ==========================================================================
def test_class_frequency_baseline_uses_training_frequencies_only():
    y_train = pd.Series([0, 0, 0, 1, 2])  # H:3/5, D:1/5, A:1/5
    proba = baselines.class_frequency_baseline(y_train, n_val=4)
    assert proba.shape == (4, 3)
    np.testing.assert_allclose(proba[0], [0.6, 0.2, 0.2])
    assert np.all(proba == proba[0])  # every row identical


def test_class_frequency_baseline_has_no_way_to_see_validation_labels():
    import inspect

    assert "y_val" not in inspect.signature(baselines.class_frequency_baseline).parameters
    y_train = pd.Series([0, 1, 2, 0])
    proba_a = baselines.class_frequency_baseline(y_train, n_val=5)
    proba_b = baselines.class_frequency_baseline(y_train, n_val=5)
    np.testing.assert_array_equal(proba_a, proba_b)


# ==========================================================================
# 20. Fold metrics use validation rows only
# ==========================================================================
def test_fold_metrics_are_computed_from_validation_rows_only(synthetic_processed_dir):
    fold = load_fold(1)
    proba = baselines.class_frequency_baseline(fold.y_train, len(fold.y_val))
    metrics = evaluation.compute_fold_metrics(
        fold=fold.fold, validation_season=fold.validation_season, y_true=fold.y_val, proba=proba
    )
    assert metrics.n_rows == len(fold.y_val)
    assert metrics.n_rows != len(fold.y_train)


# ==========================================================================
# 21. Development training code refuses 2025/26
# ==========================================================================
def test_load_fold_refuses_fold_numbers_outside_the_three_development_folds(synthetic_processed_dir):
    for bad_fold in [0, 4, 5, -1]:
        with pytest.raises(ValueError, match="sealed season must never be touched"):
            load_fold(bad_fold)


def test_development_fold_definitions_never_reference_the_sealed_season():
    for definition in datasets.DEVELOPMENT_FOLD_DEFINITIONS.values():
        assert definition["training_cutoff"] != SEALED_SEASON
        assert definition["validation_season"] != SEALED_SEASON


# ==========================================================================
# 22. Deterministic repeated fits
# ==========================================================================
def test_repeated_fits_are_deterministic_for_seeded_models(synthetic_processed_dir):
    fold = load_fold(1)
    for spec in training.MODEL_SPECS.values():
        params = spec.param_grid[0]
        proba_a = training.fit_predict(spec, params, fold)
        proba_b = training.fit_predict(spec, params, fold)
        np.testing.assert_allclose(proba_a, proba_b, atol=1e-10)


# ==========================================================================
# Model-selection rule
# ==========================================================================
def _make_result(model: str, config_id: str, proba_by_fold: list[np.ndarray], y_by_fold: list[np.ndarray]):
    fold_metrics = [
        evaluation.compute_fold_metrics(fold=i + 1, validation_season=f"fold{i + 1}", y_true=y, proba=p)
        for i, (y, p) in enumerate(zip(y_by_fold, proba_by_fold))
    ]
    log_losses = np.array([m.log_loss for m in fold_metrics])
    return training.ExperimentResult(
        model=model,
        config_id=config_id,
        params={},
        fold_metrics=fold_metrics,
        fold_proba=proba_by_fold,
        fold_y_true=y_by_fold,
        mean_log_loss=float(log_losses.mean()),
        worst_log_loss=float(log_losses.max()),
        log_loss_std=float(log_losses.std(ddof=0)),
    )


def test_select_best_configuration_picks_lowest_mean_log_loss_when_not_tied():
    n = 30
    y = np.array([0] * n)
    confident_correct = np.tile([0.9, 0.05, 0.05], (n, 1))
    confident_wrong = np.tile([0.05, 0.05, 0.9], (n, 1))

    good = _make_result("random_forest", "good", [confident_correct, confident_correct], [y, y])
    bad = _make_result("logreg", "bad", [confident_wrong, confident_wrong], [y, y])

    assert training.select_best_configuration([good, bad]).model == "random_forest"


def test_select_best_configuration_tie_break_prefers_simpler_model():
    n = 30
    y = np.array([0, 1, 2] * (n // 3))
    proba = np.tile([0.5, 0.3, 0.2], (n, 1))  # identical predictions -> exact tie

    logreg_result = _make_result("logreg", "logreg_config", [proba.copy(), proba.copy()], [y, y])
    xgboost_result = _make_result("xgboost", "xgboost_config", [proba.copy(), proba.copy()], [y, y])

    best = training.select_best_configuration([xgboost_result, logreg_result])
    assert best.model == "logreg"


def test_paired_log_loss_comparison_detects_a_clear_difference():
    n = 40
    y = np.array([0] * n)
    a = np.tile([0.9, 0.05, 0.05], (n, 1))
    b = np.tile([0.34, 0.33, 0.33], (n, 1))
    comparison = evaluation.paired_log_loss_comparison([y], [a], [b])
    assert comparison["mean_diff"] < 0  # a has lower log loss than b
    assert comparison["is_tie"] is False


def test_paired_log_loss_comparison_detects_a_tie_for_identical_predictions():
    n = 40
    y = np.array([0, 1, 2] * (n // 3))
    p = np.tile([0.5, 0.3, 0.2], (n, 1))
    comparison = evaluation.paired_log_loss_comparison([y], [p], [p])
    assert comparison["mean_diff"] == pytest.approx(0.0)
    assert comparison["is_tie"] is True


# ==========================================================================
# Baseline correctness beyond the frequency/contract checks
# ==========================================================================
def test_strength_trio_baseline_imputes_train_only(synthetic_processed_dir):
    fold = load_fold(1)
    a_proba = baselines.strength_trio_baseline(fold.X_train, fold.y_train, fold.X_val)

    mutated_val = fold.X_val.copy()
    cols = baselines.STRENGTH_TRIO_COLUMNS
    mutated_val.loc[:, cols] = mutated_val.loc[:, cols] * 1000.0 + 1_000_000.0
    b_proba = baselines.strength_trio_baseline(fold.X_train, fold.y_train, mutated_val)

    # Refitting on the SAME training data must give identical predictions on
    # the ORIGINAL (unmutated) validation rows, proving the imputer/scaler
    # inside this baseline were fit only on X_train both times.
    c_proba = baselines.strength_trio_baseline(fold.X_train, fold.y_train, fold.X_val)
    np.testing.assert_allclose(a_proba, c_proba, atol=1e-10)
    assert not np.allclose(a_proba, b_proba)
