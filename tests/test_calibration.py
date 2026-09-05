"""Tests for the calibration / ensemble meta-layer.

Most tests run on a small synthetic OOF prediction pair written to tmp CSVs
using the REAL season labels and the real 380-rows-per-season shape, so the
real `META_FOLD_DEFINITIONS` apply directly. A few integration tests read the
actual development OOF CSVs and are skipped if those are not present.

The leakage canaries here are behavioural, not merely structural: they mutate
held-out targets and assert fitted parameters are bit-identical, AND mutate
meta-training targets and assert the parameters DO change - so a canary cannot
pass vacuously.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.app.ml.calibration as cal  # noqa: E402
import backend.app.ml.evaluation as evaluation  # noqa: E402
from backend.app.ml.datasets import DEVELOPMENT_FOLD_DEFINITIONS  # noqa: E402

REAL_STAGE1 = REPO_ROOT / "reports" / "modeling" / "stage1_predictions.csv"
REAL_SCORE = REPO_ROOT / "reports" / "modeling" / "score_models_predictions.csv"
requires_real_oof = pytest.mark.skipif(
    not (REAL_STAGE1.exists() and REAL_SCORE.exists()),
    reason="development OOF prediction CSVs not present",
)

SEASON_TO_FOLD = {d["validation_season"]: f for f, d in DEVELOPMENT_FOLD_DEFINITIONS.items()}
SEASONS = ["2022_23", "2023_24", "2024_25"]
ROWS_PER_SEASON = 380


# --------------------------------------------------------------------------
# Synthetic OOF fixtures
# --------------------------------------------------------------------------
def _synthetic_rows(seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic OOF pool: 3 seasons x 380 matches, real labels."""
    rng = np.random.default_rng(seed)
    records = []
    for season_index, season in enumerate(SEASONS):
        start = pd.Timestamp(f"{season.split('_')[0]}-08-10")
        for match in range(ROWS_PER_SEASON):
            records.append(
                {
                    "fold": SEASON_TO_FOLD[season],
                    "Season": season,
                    "Date": (start + pd.Timedelta(days=match // 10)).date().isoformat(),
                    "HomeTeam": f"T{match % 20:02d}",
                    "AwayTeam": f"T{(match + 7) % 20:02d}",
                }
            )
    frame = pd.DataFrame(records)

    n = len(frame)
    logits_trio = rng.normal(0, 1.0, size=(n, 3))
    logits_dc = logits_trio * 0.8 + rng.normal(0, 0.5, size=(n, 3))  # correlated but distinct

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    p_trio = softmax(logits_trio)
    p_dc = softmax(logits_dc)
    targets = np.array([rng.choice(3, p=row) for row in p_trio])

    frame["actual_target"] = targets
    frame["actual_ftr"] = [["H", "D", "A"][t] for t in targets]
    frame["p_trio"] = list(p_trio)
    frame["p_dc"] = list(p_dc)
    return frame


def _write_synthetic_csvs(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    p_trio = np.vstack(frame["p_trio"].to_numpy())
    p_dc = np.vstack(frame["p_dc"].to_numpy())
    base = frame[["fold", "Season", "Date", "HomeTeam", "AwayTeam", "actual_target", "actual_ftr"]]

    stage1 = base.copy()
    stage1["p_home"], stage1["p_draw"], stage1["p_away"] = p_trio[:, 0], p_trio[:, 1], p_trio[:, 2]
    stage1["model"] = cal.FROZEN_OUTCOME_CONFIG
    stage1["config_id"] = cal.FROZEN_OUTCOME_CONFIG
    # A decoy config that must never be selected.
    decoy = stage1.copy()
    decoy["config_id"] = "xgboost[decoy]"
    decoy["p_home"], decoy["p_draw"], decoy["p_away"] = 0.9, 0.05, 0.05
    stage1_path = tmp_path / "stage1_predictions.csv"
    pd.concat([stage1, decoy], ignore_index=True).to_csv(stage1_path, index=False)

    score = base.copy()
    score["expected_home_goals"], score["expected_away_goals"] = 1.5, 1.2
    score["p_home"], score["p_draw"], score["p_away"] = p_dc[:, 0], p_dc[:, 1], p_dc[:, 2]
    score["most_likely_scoreline"] = "1-1"
    score["model"] = "score_model"
    score["config_id"] = cal.FROZEN_SCORELINE_CONFIG
    decoy2 = score.copy()
    decoy2["config_id"] = "poisson[decoy]"
    decoy2["p_home"], decoy2["p_draw"], decoy2["p_away"] = 0.1, 0.1, 0.8
    score_path = tmp_path / "score_models_predictions.csv"
    pd.concat([score, decoy2], ignore_index=True).to_csv(score_path, index=False)

    return stage1_path, score_path


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    return _synthetic_rows()


@pytest.fixture()
def synthetic_paths(tmp_path, synthetic_frame) -> tuple[Path, Path]:
    return _write_synthetic_csvs(tmp_path, synthetic_frame)


@pytest.fixture()
def predictions(synthetic_paths) -> cal.MetaPredictions:
    stage1_path, score_path = synthetic_paths
    return cal.load_meta_predictions(stage1_path=stage1_path, score_model_path=score_path)


# --------------------------------------------------------------------------
# 1. Loader: frozen config extraction, alignment, guards
# --------------------------------------------------------------------------
def test_only_the_two_frozen_configs_are_extracted(predictions):
    """The decoy configs written into both CSVs must never be selected."""
    assert len(predictions) == cal.EXPECTED_ROWS
    # The decoys used constant probabilities; if either had leaked in, these
    # columns would contain those constants.
    assert not np.allclose(predictions.proba_trio[:, 0], 0.9)
    assert not np.allclose(predictions.proba_dc[:, 2], 0.8)


def test_frozen_config_ids_are_the_approved_ones():
    assert cal.FROZEN_OUTCOME_CONFIG == "baseline_strength_trio"
    assert cal.FROZEN_SCORELINE_CONFIG == "dixon_coles_l2_decay"


def test_exactly_1140_aligned_rows(predictions):
    assert len(predictions) == 1140
    assert predictions.proba_trio.shape == (1140, 3)
    assert predictions.proba_dc.shape == (1140, 3)


def test_merge_is_one_to_one_and_targets_agree(predictions):
    assert predictions.frame.duplicated(cal.MERGE_KEY).sum() == 0
    assert "actual_target" in predictions.frame.columns
    assert "actual_target_dc" not in predictions.frame.columns


def test_duplicate_identity_keys_are_detected(tmp_path, synthetic_frame):
    """Overwrite one row's identity with another's, so the row COUNT stays at
    1140 and the duplicate check - not the count check - is what fires."""
    stage1_path, score_path = _write_synthetic_csvs(tmp_path, synthetic_frame)
    frame = pd.read_csv(stage1_path)
    trio_index = frame.index[frame.config_id == cal.FROZEN_OUTCOME_CONFIG]
    first, second = trio_index[0], trio_index[1]
    frame.loc[second, cal.MERGE_KEY] = frame.loc[first, cal.MERGE_KEY].to_numpy()
    frame.to_csv(stage1_path, index=False)

    assert (frame.config_id == cal.FROZEN_OUTCOME_CONFIG).sum() == cal.EXPECTED_ROWS
    with pytest.raises(ValueError, match="duplicate"):
        cal.load_meta_predictions(stage1_path=stage1_path, score_model_path=score_path)


def test_missing_match_is_detected(tmp_path, synthetic_frame):
    stage1_path, score_path = _write_synthetic_csvs(tmp_path, synthetic_frame)
    frame = pd.read_csv(score_path)
    keep = ~((frame.config_id == cal.FROZEN_SCORELINE_CONFIG) & (frame.index == frame.index[0]))
    frame[keep].to_csv(score_path, index=False)
    with pytest.raises(ValueError, match="expected 1140 rows"):
        cal.load_meta_predictions(stage1_path=stage1_path, score_model_path=score_path)


def test_target_mismatch_is_detected(tmp_path, synthetic_frame):
    stage1_path, score_path = _write_synthetic_csvs(tmp_path, synthetic_frame)
    frame = pd.read_csv(score_path)
    mask = frame.config_id == cal.FROZEN_SCORELINE_CONFIG
    first = frame.index[mask][0]
    frame.loc[first, "actual_target"] = (frame.loc[first, "actual_target"] + 1) % 3
    frame.to_csv(score_path, index=False)
    with pytest.raises(ValueError, match="mismatched actual_target"):
        cal.load_meta_predictions(stage1_path=stage1_path, score_model_path=score_path)


def test_sealed_season_is_rejected(tmp_path, synthetic_frame):
    stage1_path, score_path = _write_synthetic_csvs(tmp_path, synthetic_frame)
    for path in (stage1_path, score_path):
        frame = pd.read_csv(path)
        frame.loc[frame.index[0], "Season"] = cal.SEALED_SEASON
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="sealed-season|does not match the approved"):
        cal.load_meta_predictions(stage1_path=stage1_path, score_model_path=score_path)


def test_missing_frozen_config_raises(tmp_path, synthetic_frame):
    stage1_path, score_path = _write_synthetic_csvs(tmp_path, synthetic_frame)
    frame = pd.read_csv(stage1_path)
    frame[frame.config_id != cal.FROZEN_OUTCOME_CONFIG].to_csv(stage1_path, index=False)
    with pytest.raises(ValueError, match="not present in"):
        cal.load_meta_predictions(stage1_path=stage1_path, score_model_path=score_path)


def test_probabilities_valid_before_calibration(predictions):
    assert evaluation.validate_probabilities(predictions.proba_trio) == []
    assert evaluation.validate_probabilities(predictions.proba_dc) == []


# --------------------------------------------------------------------------
# 2. Chronological meta-fold structure
# --------------------------------------------------------------------------
def test_meta_fold_definitions_are_strictly_chronological():
    assert cal.META_FOLD_DEFINITIONS[1]["meta_train_seasons"] == ["2022_23"]
    assert cal.META_FOLD_DEFINITIONS[1]["evaluation_season"] == "2023_24"
    assert cal.META_FOLD_DEFINITIONS[2]["meta_train_seasons"] == ["2022_23", "2023_24"]
    assert cal.META_FOLD_DEFINITIONS[2]["evaluation_season"] == "2024_25"


def test_m1_uses_only_2022_23_to_fit(predictions):
    train = predictions.subset(cal.META_FOLD_DEFINITIONS[1]["meta_train_seasons"])
    assert train.seasons == ["2022_23"]
    assert len(train) == ROWS_PER_SEASON


def test_m2_uses_only_2022_23_and_2023_24_to_fit(predictions):
    train = predictions.subset(cal.META_FOLD_DEFINITIONS[2]["meta_train_seasons"])
    assert train.seasons == ["2022_23", "2023_24"]
    assert len(train) == 2 * ROWS_PER_SEASON


def test_no_future_season_appears_in_any_meta_training_set():
    order = {season: index for index, season in enumerate(SEASONS)}
    for meta_fold, definition in cal.META_FOLD_DEFINITIONS.items():
        eval_index = order[definition["evaluation_season"]]
        for season in definition["meta_train_seasons"]:
            assert order[season] < eval_index, (
                f"meta-fold {meta_fold} trains on {season}, which is not strictly "
                f"before {definition['evaluation_season']}"
            )


def test_evaluation_season_never_in_its_own_meta_training_set():
    for definition in cal.META_FOLD_DEFINITIONS.values():
        assert definition["evaluation_season"] not in definition["meta_train_seasons"]


def test_2022_23_is_the_seed_and_is_never_evaluated():
    evaluated = {d["evaluation_season"] for d in cal.META_FOLD_DEFINITIONS.values()}
    assert cal.CALIBRATION_SEED_SEASON == "2022_23"
    assert cal.CALIBRATION_SEED_SEASON not in evaluated


def test_exactly_760_rows_form_the_primary_evaluation_pool(predictions):
    result = cal.run_meta_validation(cal.CANDIDATES["raw_trio"], predictions)
    total = sum(r.n_evaluation_rows for r in result.fold_results)
    assert total == cal.EXPECTED_EVALUATION_ROWS == 760


def test_loso_protocol_is_not_implemented():
    """The anti-causal leave-one-season-out variant must not exist."""
    assert len(cal.META_FOLD_DEFINITIONS) == 2
    assert not any("loso" in name.lower() for name in dir(cal))


# --------------------------------------------------------------------------
# 3. Temperature scaling
# --------------------------------------------------------------------------
def test_temperature_one_is_the_exact_identity(predictions):
    original = predictions.proba_trio
    np.testing.assert_allclose(cal.temperature_scale(original, 1.0), original, atol=1e-12, rtol=0)


def test_temperature_outputs_sum_to_one_and_stay_in_unit_interval(predictions):
    for temperature in [0.5, 1.0, 1.5, 3.0]:
        scaled = cal.temperature_scale(predictions.proba_trio, temperature)
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0, atol=1e-12)
        assert (scaled > 0).all() and (scaled < 1).all()
        assert evaluation.validate_probabilities(scaled) == []


def test_temperature_preserves_class_ordering(predictions):
    original = predictions.proba_trio
    for temperature in [0.4, 2.5]:
        scaled = cal.temperature_scale(original, temperature)
        np.testing.assert_array_equal(original.argmax(axis=1), scaled.argmax(axis=1))


def test_higher_temperature_reduces_confidence(predictions):
    original = predictions.proba_trio
    assert cal.temperature_scale(original, 2.0).max(axis=1).mean() < original.max(axis=1).mean()
    assert cal.temperature_scale(original, 0.5).max(axis=1).mean() > original.max(axis=1).mean()


def test_temperature_hand_calculation():
    p = np.array([[0.5, 0.3, 0.2]])
    T = 2.0
    z = np.log(p[0]) / T
    expected = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    np.testing.assert_allclose(cal.temperature_scale(p, T)[0], expected, rtol=1e-12)


def test_temperature_rejects_invalid_values():
    p = np.array([[0.5, 0.3, 0.2]])
    for bad in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="finite and positive"):
            cal.temperature_scale(p, bad)


def test_fit_temperature_is_never_worse_than_the_identity(predictions):
    y, proba = predictions.y, predictions.proba_trio
    T = cal.fit_temperature(proba, y)
    fitted = cal._log_loss(cal.temperature_scale(proba, T), y)
    identity = cal._log_loss(cal.temperature_scale(proba, 1.0), y)
    assert fitted <= identity + cal.IDENTITY_TOLERANCE


def test_fit_temperature_is_deterministic(predictions):
    y, proba = predictions.y, predictions.proba_trio
    assert cal.fit_temperature(proba, y) == cal.fit_temperature(proba, y)


def test_fitted_temperature_respects_bounds(predictions):
    T = cal.fit_temperature(predictions.proba_trio, predictions.y)
    assert np.exp(cal.LOG_T_BOUNDS[0]) <= T <= np.exp(cal.LOG_T_BOUNDS[1])


# --------------------------------------------------------------------------
# 4. Convex pool and endpoint handling
# --------------------------------------------------------------------------
def test_pool_weight_one_reproduces_the_trio_exactly(predictions):
    pooled = cal.convex_pool(predictions.proba_trio, predictions.proba_dc, 1.0)
    np.testing.assert_array_equal(pooled, predictions.proba_trio)


def test_pool_weight_zero_reproduces_dixon_coles_exactly(predictions):
    pooled = cal.convex_pool(predictions.proba_trio, predictions.proba_dc, 0.0)
    np.testing.assert_array_equal(pooled, predictions.proba_dc)


def test_pool_outputs_sum_to_one(predictions):
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pooled = cal.convex_pool(predictions.proba_trio, predictions.proba_dc, w)
        np.testing.assert_allclose(pooled.sum(axis=1), 1.0, atol=1e-12)
        assert evaluation.validate_probabilities(pooled) == []


def test_pool_rejects_weights_outside_the_unit_interval(predictions):
    for bad in [-0.1, 1.1, np.nan]:
        with pytest.raises(ValueError, match="within"):
            cal.convex_pool(predictions.proba_trio, predictions.proba_dc, bad)


def test_fitted_weight_stays_within_bounds(predictions):
    w = cal.fit_pool_weight(predictions.proba_trio, predictions.proba_dc, predictions.y)
    assert cal.W_BOUNDS[0] <= w <= cal.W_BOUNDS[1]


def test_endpoints_are_explicitly_considered_not_left_to_the_optimizer(predictions):
    """When one component is strictly better everywhere, the fit must land on
    the exact endpoint - which bounded scalar optimization alone would only
    approach asymptotically."""
    y = predictions.y
    n = len(y)
    perfect = np.full((n, 3), 0.005)
    perfect[np.arange(n), y] = 0.99
    useless = np.full((n, 3), 1 / 3)

    # perfect as the trio -> optimum is exactly w = 1
    assert cal.fit_pool_weight(perfect, useless, y) == 1.0
    # perfect as the DC side -> optimum is exactly w = 0
    assert cal.fit_pool_weight(useless, perfect, y) == 0.0


def test_fit_pool_weight_is_deterministic(predictions):
    a = cal.fit_pool_weight(predictions.proba_trio, predictions.proba_dc, predictions.y)
    b = cal.fit_pool_weight(predictions.proba_trio, predictions.proba_dc, predictions.y)
    assert a == b


def test_fit_pool_weight_never_worse_than_either_endpoint(predictions):
    y, trio, dc = predictions.y, predictions.proba_trio, predictions.proba_dc
    w = cal.fit_pool_weight(trio, dc, y)
    fitted = cal._log_loss(cal.convex_pool(trio, dc, w), y)
    assert fitted <= cal._log_loss(trio, y) + cal.IDENTITY_TOLERANCE
    assert fitted <= cal._log_loss(dc, y) + cal.IDENTITY_TOLERANCE


# --------------------------------------------------------------------------
# 5. Joint (T, w)
# --------------------------------------------------------------------------
def test_joint_fit_is_never_worse_than_the_identity(predictions):
    y, trio, dc = predictions.y, predictions.proba_trio, predictions.proba_dc
    T, w = cal.fit_joint_temperature_and_weight(trio, dc, y)
    fitted = cal._log_loss(cal.convex_pool(cal.temperature_scale(trio, T), dc, w), y)
    identity = cal._log_loss(trio, y)  # (T=1, w=1) is exactly the raw trio
    assert fitted <= identity + cal.IDENTITY_TOLERANCE


def test_joint_identity_point_equals_the_raw_trio(predictions):
    trio = predictions.proba_trio
    identity = cal.convex_pool(cal.temperature_scale(trio, 1.0), predictions.proba_dc, 1.0)
    np.testing.assert_allclose(identity, trio, atol=1e-12, rtol=0)


def test_joint_fit_respects_bounds_and_is_deterministic(predictions):
    y, trio, dc = predictions.y, predictions.proba_trio, predictions.proba_dc
    T1, w1 = cal.fit_joint_temperature_and_weight(trio, dc, y)
    T2, w2 = cal.fit_joint_temperature_and_weight(trio, dc, y)
    assert (T1, w1) == (T2, w2)
    assert np.exp(cal.LOG_T_BOUNDS[0]) <= T1 <= np.exp(cal.LOG_T_BOUNDS[1])
    assert cal.W_BOUNDS[0] <= w1 <= cal.W_BOUNDS[1]


# --------------------------------------------------------------------------
# 6. Leakage canaries (behavioural, with a sensitivity control)
# --------------------------------------------------------------------------
def _fit_on(predictions: cal.MetaPredictions, seasons, candidate_name: str) -> dict:
    train = predictions.subset(seasons)
    return cal.CANDIDATES[candidate_name].fit(train.proba_trio, train.proba_dc, train.y)


def _mutate_targets(predictions: cal.MetaPredictions, seasons) -> cal.MetaPredictions:
    frame = predictions.frame.copy()
    mask = frame["Season"].isin(list(seasons))
    frame.loc[mask, "actual_target"] = (frame.loc[mask, "actual_target"] + 1) % 3
    return cal.MetaPredictions(frame)


@pytest.mark.parametrize("candidate", ["temperature_trio", "pool_raw_trio", "pool_calibrated_trio"])
def test_mutating_held_out_targets_cannot_change_fitted_parameters(predictions, candidate):
    """M1 fits on 2022_23; mutating the held-out 2023_24/2024_25 targets must
    leave the fitted parameters bit-identical."""
    train_seasons = cal.META_FOLD_DEFINITIONS[1]["meta_train_seasons"]
    before = _fit_on(predictions, train_seasons, candidate)
    mutated = _mutate_targets(predictions, ["2023_24", "2024_25"])
    after = _fit_on(mutated, train_seasons, candidate)
    assert before == after


@pytest.mark.parametrize("candidate", ["temperature_trio", "pool_raw_trio", "pool_calibrated_trio"])
def test_mutating_meta_training_targets_does_change_parameters(predictions, candidate):
    """Sensitivity control: proves the canary above is not vacuous."""
    train_seasons = cal.META_FOLD_DEFINITIONS[1]["meta_train_seasons"]
    before = _fit_on(predictions, train_seasons, candidate)
    mutated = _mutate_targets(predictions, train_seasons)
    after = _fit_on(mutated, train_seasons, candidate)
    assert before != after


def test_mutating_held_out_targets_cannot_change_held_out_predictions(predictions):
    """A candidate's held-out probabilities depend only on the fitted params and
    the held-out FEATURES (base-model probabilities), never on held-out labels."""
    train_seasons = cal.META_FOLD_DEFINITIONS[1]["meta_train_seasons"]
    evaluate = predictions.subset(["2023_24"])
    params = _fit_on(predictions, train_seasons, "temperature_trio")
    proba_a = cal.CANDIDATES["temperature_trio"].apply(evaluate.proba_trio, evaluate.proba_dc, params)

    mutated = _mutate_targets(predictions, ["2023_24"])
    evaluate_b = mutated.subset(["2023_24"])
    params_b = _fit_on(mutated, train_seasons, "temperature_trio")
    proba_b = cal.CANDIDATES["temperature_trio"].apply(evaluate_b.proba_trio, evaluate_b.proba_dc, params_b)

    np.testing.assert_array_equal(proba_a, proba_b)


# --------------------------------------------------------------------------
# 7. Candidates, comparison fairness, selection rule
# --------------------------------------------------------------------------
def test_exactly_four_precommitted_candidates():
    assert list(cal.CANDIDATES) == [
        "raw_trio",
        "temperature_trio",
        "pool_raw_trio",
        "pool_calibrated_trio",
    ]
    assert cal.INCUMBENT_CANDIDATE == "raw_trio"
    assert cal.CANDIDATES["raw_trio"].n_params == 0
    assert cal.CANDIDATES["temperature_trio"].n_params == 1
    assert cal.CANDIDATES["pool_raw_trio"].n_params == 1
    assert cal.CANDIDATES["pool_calibrated_trio"].n_params == 2


def test_raw_trio_candidate_is_the_untouched_trio(predictions):
    result = cal.run_meta_validation(cal.CANDIDATES["raw_trio"], predictions)
    for fold_result in result.fold_results:
        evaluate = predictions.subset([fold_result.evaluation_season])
        np.testing.assert_array_equal(fold_result.proba, evaluate.proba_trio)
        assert fold_result.fitted_params == {}


def test_incumbent_and_challengers_are_scored_on_identical_rows(predictions):
    results = {
        name: cal.run_meta_validation(candidate, predictions)
        for name, candidate in cal.CANDIDATES.items()
    }
    incumbent = results[cal.INCUMBENT_CANDIDATE]
    for name, result in results.items():
        for a, b in zip(result.fold_results, incumbent.fold_results):
            assert a.evaluation_season == b.evaluation_season
            assert a.n_evaluation_rows == b.n_evaluation_rows
            np.testing.assert_array_equal(a.y_true, b.y_true), name


def test_selection_retains_incumbent_when_no_challenger_qualifies(predictions):
    results = [cal.run_meta_validation(c, predictions) for c in cal.CANDIDATES.values()]
    selection = cal.select_best_candidate(results)
    assert selection.selected in cal.CANDIDATES
    assert selection.n_evaluation_rows == 760
    for name, verdict in selection.reasons.items():
        assert set(verdict) >= {
            "lower_mean_log_loss",
            "paired_improvement_not_a_tie",
            "improves_every_evaluation_season",
            "no_worst_season_regression",
            "passes_all_conditions",
        }


def test_selection_requires_all_four_conditions():
    """A challenger that is better on average but worse in one season must lose."""
    y = np.array([0, 1, 2] * 50)

    def make(name, per_season_ll, worst, mean_ll, proba, n_params=1):
        folds = []
        for season, _ in per_season_ll.items():
            metrics = evaluation.compute_fold_metrics(
                fold=1, validation_season=season, y_true=y, proba=proba
            )
            folds.append(
                cal.MetaFoldResult(
                    meta_fold=1, meta_train_seasons=["2022_23"], evaluation_season=season,
                    n_meta_train_rows=380, n_evaluation_rows=len(y), fitted_params={},
                    metrics=metrics, proba=proba, y_true=y,
                )
            )
        return cal.MetaCandidateResult(
            name=name, n_params=n_params, fold_results=folds,
            mean_log_loss=mean_ll, worst_log_loss=worst,
        )

    flat = np.tile([0.34, 0.33, 0.33], (len(y), 1))
    incumbent = make("raw_trio", {"2023_24": 1.0, "2024_25": 1.0}, 1.0, 1.0, flat, n_params=0)
    # Reported mean is lower, but the PREDICTIONS are identical, so the paired
    # comparison is an exact tie and condition 2 must fail.
    challenger = make("temperature_trio", {"2023_24": 0.9, "2024_25": 0.9}, 0.9, 0.9, flat)

    selection = cal.select_best_candidate([incumbent, challenger])
    verdict = selection.reasons["temperature_trio"]
    assert verdict["lower_mean_log_loss"] is True
    assert verdict["paired_improvement_not_a_tie"] is False
    assert verdict["passes_all_conditions"] is False
    assert selection.selected == "raw_trio"
    assert selection.incumbent_retained is True


def test_accuracy_brier_and_f1_are_reported_but_never_decide(predictions):
    result = cal.run_meta_validation(cal.CANDIDATES["raw_trio"], predictions)
    metrics = result.fold_results[0].metrics
    assert metrics.accuracy is not None
    assert metrics.brier_score is not None
    assert metrics.macro_f1 is not None
    # The selection verdict keys contain no accuracy/Brier/F1 condition.
    results = [cal.run_meta_validation(c, predictions) for c in cal.CANDIDATES.values()]
    selection = cal.select_best_candidate(results)
    for verdict in selection.reasons.values():
        keys = " ".join(verdict.keys()).lower()
        assert "accuracy" not in keys and "brier" not in keys and "f1" not in keys


# --------------------------------------------------------------------------
# 8. Final meta-parameter fit gating
# --------------------------------------------------------------------------
def test_final_fit_requires_a_frozen_selection(predictions):
    """A raw string or dict must not be accepted - the final fit is
    structurally impossible before selection."""
    for not_a_selection in ["temperature_trio", {"selected": "temperature_trio"}, None]:
        with pytest.raises(TypeError, match="must never precede method selection"):
            cal.fit_final_meta_parameters(not_a_selection, predictions)


def test_final_fit_returns_identity_when_incumbent_is_retained(predictions):
    selection = cal.MetaSelection(
        selected="raw_trio", incumbent_retained=True, reasons={},
        incumbent_mean_log_loss=1.0, n_evaluation_rows=760,
    )
    assert cal.fit_final_meta_parameters(selection, predictions) == {"T": 1.0, "w": 1.0}


def test_final_fit_uses_all_1140_rows_when_a_challenger_wins(predictions):
    selection = cal.MetaSelection(
        selected="temperature_trio", incumbent_retained=False, reasons={},
        incumbent_mean_log_loss=1.0, n_evaluation_rows=760,
    )
    params = cal.fit_final_meta_parameters(selection, predictions)
    assert set(params) == {"T"}
    expected = cal.fit_temperature(predictions.proba_trio, predictions.y)
    assert params["T"] == expected  # fitted on all 1,140 rows, not a subset


def test_final_fit_rejects_a_partial_development_pool(predictions):
    selection = cal.MetaSelection(
        selected="temperature_trio", incumbent_retained=False, reasons={},
        incumbent_mean_log_loss=1.0, n_evaluation_rows=760,
    )
    with pytest.raises(ValueError, match="expects all 1140"):
        cal.fit_final_meta_parameters(selection, predictions.subset(["2022_23"]))


# --------------------------------------------------------------------------
# 9. Invariants: scoreline consistency, evaluation reuse, class order
# --------------------------------------------------------------------------
def test_dixon_coles_probabilities_are_never_modified(predictions):
    """Temperature is applied only to the trio; DC passes through untouched."""
    before = predictions.proba_dc.copy()
    for candidate in cal.CANDIDATES.values():
        cal.run_meta_validation(candidate, predictions)
    np.testing.assert_array_equal(predictions.proba_dc, before)


def test_dc_columns_match_the_source_csv_after_the_pipeline(synthetic_paths):
    stage1_path, score_path = synthetic_paths
    predictions = cal.load_meta_predictions(stage1_path=stage1_path, score_model_path=score_path)
    for candidate in cal.CANDIDATES.values():
        cal.run_meta_validation(candidate, predictions)

    source = pd.read_csv(score_path)
    source = source[source.config_id == cal.FROZEN_SCORELINE_CONFIG]
    source = source.sort_values(cal.MERGE_KEY).reset_index(drop=True)
    np.testing.assert_allclose(
        predictions.proba_dc, source[["p_home", "p_draw", "p_away"]].to_numpy(), rtol=0, atol=0
    )


def test_score_models_and_training_modules_are_not_imported():
    """The meta-layer reads CSVs; it must never import a module that could
    refit a base model."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cal))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "backend.app.ml.score_models",
        "backend.app.ml.training",
        "backend.app.ml.baselines",
    }
    assert not (imported & forbidden), f"calibration.py must not import {imported & forbidden}"
    assert "backend.app.ml.evaluation" in imported  # but evaluation.py IS reused


def test_evaluation_functions_are_reused_by_identity():
    assert cal.compute_fold_metrics is evaluation.compute_fold_metrics
    assert cal.validate_probabilities is evaluation.validate_probabilities
    assert cal.paired_log_loss_comparison is evaluation.paired_log_loss_comparison


def test_class_order_is_h_d_a(predictions):
    assert cal.EXPECTED_CLASSES == (0, 1, 2)
    assert cal.CLASS_NAMES == ("H", "D", "A")
    assert cal.TRIO_PROBA_COLUMNS == ["p_home_trio", "p_draw_trio", "p_away_trio"]
    assert cal.DC_PROBA_COLUMNS == ["p_home_dc", "p_draw_dc", "p_away_dc"]


def test_per_class_reliability_shape(predictions):
    table = cal.per_class_reliability(predictions.y, predictions.proba_trio)
    assert set(table) == {"H", "D", "A"}
    for rows in table.values():
        assert sum(row["n"] for row in rows) == len(predictions)
        for row in rows:
            assert 0.0 <= row["mean_predicted"] <= 1.0
            assert 0.0 <= row["observed_rate"] <= 1.0


# --------------------------------------------------------------------------
# 10. Real-data integration
# --------------------------------------------------------------------------
@requires_real_oof
def test_real_oof_pool_loads_with_1140_aligned_rows():
    predictions = cal.load_meta_predictions()
    assert len(predictions) == 1140
    assert predictions.seasons == ["2022_23", "2023_24", "2024_25"]
    assert cal.SEALED_SEASON not in predictions.seasons
    assert evaluation.validate_probabilities(predictions.proba_trio) == []
    assert evaluation.validate_probabilities(predictions.proba_dc) == []


@requires_real_oof
def test_real_evaluation_pool_is_760_rows():
    predictions = cal.load_meta_predictions()
    assert len(predictions.subset(cal.EVALUATION_SEASONS)) == 760
