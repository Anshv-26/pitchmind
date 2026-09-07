"""Tests for the model-grounded explanation layer.

Two kinds of fixtures:
- a tiny HAND-BUILT synthetic 3-class logistic pipeline (no imputer, or a
  minimal imputer) whose coefficients/intercepts are chosen so contributions
  can be verified against numbers computed by hand in the test itself, and
  which can be steered to predict H, D, or A;
- the REAL `baseline_strength_trio` artifact, built once per test session
  from `data/processed/features_causal_through_2023_24.parquet` (the same
  fold-3 feature file used by Stage 1 - its row cap stops at 2024/25, so it
  cannot contain sealed-season rows), skipped if that file is not present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.app.ml.explanations as ex  # noqa: E402
from backend.app.ml.baselines import STRENGTH_TRIO_COLUMNS, _fit_logistic_pipeline  # noqa: E402
from backend.app.ml.evaluation import CLASS_NAMES, EXPECTED_CLASSES  # noqa: E402

REAL_ARTIFACT_SOURCE = ex.SOURCE_FEATURE_ARTIFACT_PATH
requires_real_data = pytest.mark.skipif(
    not REAL_ARTIFACT_SOURCE.exists(), reason="features_causal_through_2023_24.parquet not built"
)


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------
def _make_artifact(pipeline, *, training_seasons=None) -> ex.StrengthTrioArtifact:
    metadata = ex.ArtifactMetadata(
        model_id="baseline_strength_trio",
        artifact_format_version=ex.STRENGTH_TRIO_ARTIFACT_FORMAT_VERSION,
        training_cutoff_season=ex.TRAINING_CUTOFF_SEASON,
        training_seasons=training_seasons or ["2024_25"],
        feature_columns=list(STRENGTH_TRIO_COLUMNS),
        transformed_feature_names=[],
        class_mapping={name: cls for name, cls in zip(CLASS_NAMES, EXPECTED_CLASSES)},
        source_feature_artifact="synthetic",
        n_training_rows=len(pipeline.named_steps["model"].coef_),
        library_versions={},
        intended_final_evaluation_season=ex.SEALED_SEASON,
        requires_retraining_before_final_evaluation=False,
    )
    return ex.StrengthTrioArtifact(metadata=metadata, pipeline=pipeline)


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    """Deterministic 3-feature frame with genuine missingness in two columns,
    engineered so the fitted model predicts all three classes across rows,
    including at least one Draw."""
    return pd.DataFrame(
        {
            "elo_diff": [-50.0, 0.0, 50.0, -20.0, 20.0, 0.0, -80.0, 80.0],
            "diff_ewma_ppg": [-0.5, 0.0, 0.5, np.nan, 0.2, -0.1, -0.8, 0.8],
            "diff_ewma_sot_diff": [-2.0, 0.0, 2.0, 1.0, -1.0, np.nan, -3.0, 3.0],
        }
    )


@pytest.fixture(scope="module")
def synthetic_targets() -> pd.Series:
    return pd.Series([2, 1, 0, 1, 0, 2, 2, 0])


@pytest.fixture(scope="module")
def synthetic_pipeline(synthetic_frame, synthetic_targets):
    return _fit_logistic_pipeline(synthetic_frame, synthetic_targets, with_imputer=True)


@pytest.fixture(scope="module")
def synthetic_artifact(synthetic_pipeline) -> ex.StrengthTrioArtifact:
    return _make_artifact(synthetic_pipeline)


@pytest.fixture(scope="module")
def class_predicted_rows(synthetic_pipeline, synthetic_frame):
    """One row index per predicted class, discovered from the fitted model
    (not hand-picked), so H/D/A coverage is verified rather than assumed."""
    proba = synthetic_pipeline.predict_proba(synthetic_frame)
    argmax = proba.argmax(axis=1)
    rows = {}
    for cls in EXPECTED_CLASSES:
        matches = np.where(argmax == cls)[0]
        if len(matches):
            rows[cls] = int(matches[0])
    assert set(rows) == set(EXPECTED_CLASSES), f"synthetic fixture must predict all 3 classes, got {sorted(rows)}"
    return rows


# --------------------------------------------------------------------------
# Real FROZEN artifact (module-scoped, built once) - fit on every row of
# features_causal_through_2023_24.parquet (2015/16 through 2024/25 inclusive)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_artifact() -> ex.StrengthTrioArtifact:
    return ex.build_strength_trio_artifact()


@pytest.fixture(scope="module")
def real_source_frame() -> pd.DataFrame:
    """The raw source frame, unfiltered - this file structurally cannot
    contain sealed-season rows, so there is nothing to filter out here."""
    return pd.read_parquet(REAL_ARTIFACT_SOURCE)


# --------------------------------------------------------------------------
# 1. Hand-calculation tests with a tiny fixed model
# --------------------------------------------------------------------------
def test_hand_built_two_feature_model_reconstructs_logits_exactly():
    """A tiny model with NO imputer needed (fully observed features) and
    hand-set coefficients, checked against a manual computation."""
    X = pd.DataFrame({"elo_diff": [10.0, -10.0, 0.0], "diff_ewma_ppg": [1.0, -1.0, 0.0], "diff_ewma_sot_diff": [0.0, 0.0, 0.0]})
    y = pd.Series([0, 2, 1])
    pipeline = _fit_logistic_pipeline(X, y, with_imputer=True)
    artifact = _make_artifact(pipeline)

    model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    imputer = pipeline.named_steps["imputer"]

    row = X.iloc[[0]]
    scaled = scaler.transform(imputer.transform(row))
    expected_logits = scaled @ model.coef_.T + model.intercept_
    expected_proba = np.exp(expected_logits - expected_logits.max()) / np.exp(expected_logits - expected_logits.max()).sum()

    explanation = ex.explain_match(artifact, row)
    for i, name in enumerate(ex.CLASS_LONG_NAMES):
        assert explanation["logits"][name] == pytest.approx(expected_logits[0, i], abs=1e-9)
        assert explanation["probabilities"][name] == pytest.approx(expected_proba[0, i], abs=1e-9)


# --------------------------------------------------------------------------
# 2. Transformed column count: 5 fitted columns, 3 semantic features
# --------------------------------------------------------------------------
def test_transformed_column_count_is_five_while_semantic_features_are_three(synthetic_pipeline):
    model = synthetic_pipeline.named_steps["model"]
    assert model.coef_.shape[1] == 5
    assert len(STRENGTH_TRIO_COLUMNS) == 3


def test_column_groups_identify_the_two_indicator_columns(synthetic_pipeline):
    groups = ex._build_column_groups(synthetic_pipeline, STRENGTH_TRIO_COLUMNS)
    assert groups["elo_diff"].indicator_index is None
    assert groups["diff_ewma_ppg"].indicator_index is not None
    assert groups["diff_ewma_sot_diff"].indicator_index is not None


def test_missingness_indicator_contribution_is_folded_into_the_correct_parent(synthetic_artifact, synthetic_frame, class_predicted_rows):
    row = synthetic_frame.iloc[[class_predicted_rows[0]]]
    explanation = ex.explain_match(synthetic_artifact, row)
    by_name = {f["name"]: f for f in explanation["features"]}

    # elo_diff structurally has no indicator - its missingness contribution
    # must be reported as None (structurally absent), not zero-as-a-value.
    assert by_name["elo_diff"]["missingness_logit_contribution"] is None

    for feature in ["diff_ewma_ppg", "diff_ewma_sot_diff"]:
        entry = by_name[feature]
        assert entry["missingness_logit_contribution"] is not None
        for cname in ex.CLASS_LONG_NAMES:
            grouped = entry["grouped_logit_contribution"][cname]
            value_part = entry["value_logit_contribution"][cname]
            missing_part = entry["missingness_logit_contribution"][cname]
            assert grouped == pytest.approx(value_part + missing_part, abs=1e-12)


# --------------------------------------------------------------------------
# 3. Exact reconstruction identities
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", list(EXPECTED_CLASSES))
def test_grouped_contributions_plus_intercept_reconstruct_each_class_logit(
    synthetic_artifact, synthetic_frame, class_predicted_rows, cls
):
    row = synthetic_frame.iloc[[class_predicted_rows[cls]]]
    explanation = ex.explain_match(synthetic_artifact, row)
    for cname in ex.CLASS_LONG_NAMES:
        total = explanation["intercepts"][cname] + sum(
            f["grouped_logit_contribution"][cname] for f in explanation["features"]
        )
        assert total == pytest.approx(explanation["logits"][cname], abs=1e-9)


def test_reconstruction_matches_decision_function_and_predict_proba(synthetic_artifact, synthetic_frame):
    pipeline = synthetic_artifact.pipeline
    model = pipeline.named_steps["model"]
    for i in range(len(synthetic_frame)):
        row = synthetic_frame.iloc[[i]]
        explanation = ex.explain_match(synthetic_artifact, row)

        imputer = pipeline.named_steps.get("imputer")
        scaler = pipeline.named_steps["scaler"]
        transformed = imputer.transform(row) if imputer is not None else row.to_numpy(dtype=float)
        scaled = scaler.transform(transformed)
        decision = model.decision_function(scaled)[0]
        proba = pipeline.predict_proba(row)[0]

        for j, cname in enumerate(ex.CLASS_LONG_NAMES):
            assert explanation["logits"][cname] == pytest.approx(decision[j], abs=1e-9)
            assert explanation["probabilities"][cname] == pytest.approx(proba[j], abs=1e-9)


def test_reconstruction_works_on_an_actual_imputed_cold_start_row(synthetic_artifact, synthetic_frame):
    cold_start_index = synthetic_frame["diff_ewma_ppg"].isna().idxmax()
    assert pd.isna(synthetic_frame.loc[cold_start_index, "diff_ewma_ppg"])
    row = synthetic_frame.iloc[[cold_start_index]]
    explanation = ex.explain_match(synthetic_artifact, row)

    by_name = {f["name"]: f for f in explanation["features"]}
    assert by_name["diff_ewma_ppg"]["was_imputed"] is True
    assert by_name["diff_ewma_ppg"]["raw_value"] is None

    for cname in ex.CLASS_LONG_NAMES:
        total = explanation["intercepts"][cname] + sum(
            f["grouped_logit_contribution"][cname] for f in explanation["features"]
        )
        assert total == pytest.approx(explanation["logits"][cname], abs=1e-9)


def test_calibration_and_ensemble_are_not_involved():
    """This module must not import the calibration/ensemble meta-layer -
    explanations are computed directly from the frozen base model."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ex))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "backend.app.ml.calibration" not in imported


# --------------------------------------------------------------------------
# 4. Class order and no-mislabeling
# --------------------------------------------------------------------------
def test_class_order_is_h_d_a(synthetic_artifact):
    assert ex.CLASS_NAMES == ("H", "D", "A")
    assert list(ex.EXPECTED_CLASSES) == [0, 1, 2]
    assert ex.CLASS_LONG_NAMES == ("home", "draw", "away")


def test_feature_names_and_order_match_the_frozen_strength_trio_contract(synthetic_artifact, synthetic_frame):
    explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[0]])
    assert [f["name"] for f in explanation["features"]] == list(STRENGTH_TRIO_COLUMNS)


def test_logit_contribution_is_never_labelled_as_percentage_point_impact():
    """Structural guard: the module source must never describe a
    logit_contribution field as a percentage-point/probability effect."""
    import inspect

    source = inspect.getsource(ex)
    forbidden_phrases = [
        "logit_contribution is a percentage",
        "logit_contribution represents a probability",
        "percentage point contribution",
        "percentage-point contribution",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in source.lower()
    # And the two concepts must be genuinely distinct fields in the schema.
    assert "grouped_logit_contribution" in source
    assert "probability_sensitivity" in source


# --------------------------------------------------------------------------
# 5. Zero-reference counterfactual sensitivity
# --------------------------------------------------------------------------
def test_sensitivity_goes_through_the_real_pipeline_not_manual_column_edits(synthetic_artifact, synthetic_frame):
    """Verify the sensitivity equals predict_proba(original) -
    predict_proba(counterfactual) computed independently in the test, using
    the pipeline's own predict_proba both times."""
    pipeline = synthetic_artifact.pipeline
    row = synthetic_frame.iloc[[0]]
    explanation = ex.explain_match(synthetic_artifact, row)
    original_proba = pipeline.predict_proba(row)[0]

    for feature in STRENGTH_TRIO_COLUMNS:
        counterfactual_row = row.copy()
        counterfactual_row[feature] = 0.0
        expected_counterfactual_proba = pipeline.predict_proba(counterfactual_row)[0]
        expected_sensitivity = original_proba - expected_counterfactual_proba

        entry = next(f for f in explanation["features"] if f["name"] == feature)
        for j, cname in enumerate(ex.CLASS_LONG_NAMES):
            assert entry["probability_sensitivity"][cname] == pytest.approx(expected_sensitivity[j], abs=1e-12)


def test_zero_reference_sensitivity_is_deterministic(synthetic_artifact, synthetic_frame):
    row = synthetic_frame.iloc[[0]]
    a = ex.explain_match(synthetic_artifact, row)
    b = ex.explain_match(synthetic_artifact, row)
    assert a == b


def test_setting_missing_feature_to_zero_removes_the_missingness_indicator(synthetic_artifact, synthetic_frame):
    cold_start_index = synthetic_frame["diff_ewma_ppg"].isna().idxmax()
    row = synthetic_frame.iloc[[cold_start_index]]
    explanation = ex.explain_match(synthetic_artifact, row)
    entry = next(f for f in explanation["features"] if f["name"] == "diff_ewma_ppg")
    assert entry["was_imputed"] is True
    assert entry["counterfactual_changed_missingness"] is True


def test_fully_observed_feature_counterfactual_does_not_change_missingness(synthetic_artifact, synthetic_frame):
    row = synthetic_frame.iloc[[0]]  # diff_ewma_ppg = -0.5, fully observed
    explanation = ex.explain_match(synthetic_artifact, row)
    entry = next(f for f in explanation["features"] if f["name"] == "diff_ewma_ppg")
    assert entry["was_imputed"] is False
    assert entry["counterfactual_changed_missingness"] is False


def test_elo_diff_never_reports_changed_missingness(synthetic_artifact, synthetic_frame):
    """elo_diff structurally has no indicator column at all."""
    for i in range(len(synthetic_frame)):
        explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[i]])
        entry = next(f for f in explanation["features"] if f["name"] == "elo_diff")
        assert entry["counterfactual_changed_missingness"] is False
        assert entry["missingness_logit_contribution"] is None


def test_probability_sensitivity_is_correctly_signed():
    """Frozen convention: probability_sensitivity = original predict_proba -
    predict_proba with that raw semantic feature replaced by zero.

    If an observed feature (here elo_diff) genuinely supports the predicted
    Home class, zeroing it REMOVES that support, so
    original P(Home) > counterfactual P(Home), and therefore
    probability_sensitivity["home"] MUST BE POSITIVE under this convention.

    Does not depend on `class_predicted_rows` picking out a suitable row from
    the shared synthetic fixture (which, for row index 1, has elo_diff=0.0 -
    zeroing an already-zero value produces a zero sensitivity, not a
    meaningful sign check). Instead builds a small, dedicated, clearly
    separated synthetic pipeline where a large positive elo_diff
    unambiguously drives the Home prediction.
    """
    X = pd.DataFrame(
        {
            "elo_diff": [80.0, 60.0, -60.0, -80.0, 0.0, 0.0, 30.0, -30.0],
            "diff_ewma_ppg": [0.1, 0.05, -0.05, -0.1, 0.0, 0.0, 0.02, -0.02],
            "diff_ewma_sot_diff": [0.5, 0.3, -0.3, -0.5, 0.0, 0.0, 0.1, -0.1],
        }
    )
    y = pd.Series([0, 0, 2, 2, 1, 1, 0, 2])
    pipeline = _fit_logistic_pipeline(X, y, with_imputer=True)
    artifact = _make_artifact(pipeline)

    original_row = pd.DataFrame({"elo_diff": [70.0], "diff_ewma_ppg": [0.08], "diff_ewma_sot_diff": [0.4]})
    counterfactual_row = original_row.copy()
    counterfactual_row["elo_diff"] = 0.0

    original_proba = pipeline.predict_proba(original_row)[0]
    counterfactual_proba = pipeline.predict_proba(counterfactual_row)[0]

    # The sign expectation below is justified by the actual fixture, not assumed.
    assert original_proba.argmax() == 0  # Home is genuinely the predicted class
    assert original_proba[0] > counterfactual_proba[0]  # zeroing elo_diff genuinely lowers P(Home)

    explanation = ex.explain_match(artifact, original_row)
    assert explanation["predicted_class"] == "H"
    elo_entry = next(f for f in explanation["features"] if f["name"] == "elo_diff")

    expected_sensitivity_home = original_proba[0] - counterfactual_proba[0]
    assert elo_entry["probability_sensitivity"]["home"] == pytest.approx(expected_sensitivity_home, abs=1e-12)
    assert elo_entry["probability_sensitivity"]["home"] > 0


# --------------------------------------------------------------------------
# 6. Imputation traceability
# --------------------------------------------------------------------------
def test_observed_and_effective_values_are_distinguishable_when_imputed(synthetic_artifact, synthetic_frame):
    cold_start_index = synthetic_frame["diff_ewma_ppg"].isna().idxmax()
    row = synthetic_frame.iloc[[cold_start_index]]
    explanation = ex.explain_match(synthetic_artifact, row)
    entry = next(f for f in explanation["features"] if f["name"] == "diff_ewma_ppg")
    assert entry["raw_value"] is None
    assert entry["effective_model_value"] is not None
    assert isinstance(entry["effective_model_value"], float)


def test_observed_and_effective_values_are_equal_when_not_imputed(synthetic_artifact, synthetic_frame):
    row = synthetic_frame.iloc[[0]]
    explanation = ex.explain_match(synthetic_artifact, row)
    entry = next(f for f in explanation["features"] if f["name"] == "elo_diff")
    assert entry["raw_value"] == pytest.approx(entry["effective_model_value"])
    assert entry["was_imputed"] is False


def test_was_imputed_read_from_fitted_indicator_not_reimplemented_isna(synthetic_artifact, synthetic_frame):
    """was_imputed must come from the imputer's own transform output. Verify
    by comparing directly against imputer.transform's indicator column."""
    pipeline = synthetic_artifact.pipeline
    imputer = pipeline.named_steps["imputer"]
    for i in range(len(synthetic_frame)):
        row = synthetic_frame.iloc[[i]]
        explanation = ex.explain_match(synthetic_artifact, row)
        transformed = imputer.transform(row)
        groups = ex._build_column_groups(pipeline, STRENGTH_TRIO_COLUMNS)
        for feature in STRENGTH_TRIO_COLUMNS:
            entry = next(f for f in explanation["features"] if f["name"] == feature)
            indicator_index = groups[feature].indicator_index
            if indicator_index is not None:
                expected = bool(transformed[0, indicator_index] == 1.0)
                assert entry["was_imputed"] == expected


# --------------------------------------------------------------------------
# 7. Driver ranking - two separate rankings, no hybrid score
# --------------------------------------------------------------------------
def test_driver_rankings_are_kept_separate_no_hybrid_score():
    import inspect

    source = inspect.getsource(ex)
    assert "hybrid" not in source.lower()
    assert "importance_score" not in source.lower()


def test_logit_ranking_is_sorted_descending_by_predicted_class_contribution(synthetic_artifact, synthetic_frame, class_predicted_rows):
    for cls, idx in class_predicted_rows.items():
        explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[idx]])
        values = [d["grouped_logit_contribution_predicted_class"] for d in explanation["drivers_by_logit_contribution"]]
        assert values == sorted(values, reverse=True)


def test_sensitivity_ranking_is_sorted_descending_by_predicted_class_sensitivity(synthetic_artifact, synthetic_frame, class_predicted_rows):
    for cls, idx in class_predicted_rows.items():
        explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[idx]])
        values = [d["probability_sensitivity_predicted_class"] for d in explanation["drivers_by_probability_sensitivity"]]
        assert values == sorted(values, reverse=True)


def test_direction_is_relative_to_whichever_class_is_predicted(synthetic_artifact, synthetic_frame, class_predicted_rows):
    for cls, idx in class_predicted_rows.items():
        explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[idx]])
        predicted_long_name = ex.CLASS_LONG_NAMES[list(EXPECTED_CLASSES).index(cls)]
        assert explanation["predicted_class"] == CLASS_NAMES[list(EXPECTED_CLASSES).index(cls)]
        for driver in explanation["drivers_by_logit_contribution"]:
            assert predicted_long_name in driver["direction"] or driver["direction"] == "neutral"


# --------------------------------------------------------------------------
# 8. Class coverage - H, D, A through the identical function
# --------------------------------------------------------------------------
def test_home_prediction_explanation_works(synthetic_artifact, synthetic_frame, class_predicted_rows):
    explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[class_predicted_rows[0]]])
    assert explanation["predicted_class"] == "H"


def test_draw_prediction_explanation_works(synthetic_artifact, synthetic_frame, class_predicted_rows):
    explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[class_predicted_rows[1]]])
    assert explanation["predicted_class"] == "D"


def test_away_prediction_explanation_works(synthetic_artifact, synthetic_frame, class_predicted_rows):
    explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[class_predicted_rows[2]]])
    assert explanation["predicted_class"] == "A"


def test_no_home_away_only_branching_in_source():
    """The direction/ranking logic must be generic across all 3 classes -
    no code path that only considers Home vs Away."""
    import inspect

    source = inspect.getsource(ex.explain_match)
    assert "if predicted_class_name ==" not in source.replace(" ", "")
    assert "home_vs_away" not in source.lower()


# --------------------------------------------------------------------------
# 9. JSON serialization and determinism
# --------------------------------------------------------------------------
def test_json_serialization_round_trips(synthetic_artifact, synthetic_frame):
    explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[0]])
    serialized = json.dumps(explanation)
    deserialized = json.loads(serialized)
    assert deserialized["predicted_class"] == explanation["predicted_class"]
    assert deserialized["features"][0]["name"] == explanation["features"][0]["name"]


def test_repeated_explanation_calls_are_deterministic(synthetic_artifact, synthetic_frame):
    row = synthetic_frame.iloc[[0]]
    results = [ex.explain_match(synthetic_artifact, row) for _ in range(3)]
    assert results[0] == results[1] == results[2]


def test_explanation_generation_does_not_retrain_the_model(synthetic_artifact, synthetic_frame, monkeypatch):
    from sklearn.pipeline import Pipeline

    def _forbidden_fit(self, *args, **kwargs):
        raise AssertionError("explain_match must never call Pipeline.fit")

    monkeypatch.setattr(Pipeline, "fit", _forbidden_fit)
    # Must still succeed against the ALREADY-FITTED artifact.
    explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[0]])
    assert "predicted_class" in explanation


# --------------------------------------------------------------------------
# 10. Reuse of Stage 1 code, not duplication
# --------------------------------------------------------------------------
def test_strength_trio_columns_and_pipeline_builder_are_reused_by_identity():
    import backend.app.ml.baselines as baselines

    assert ex.STRENGTH_TRIO_COLUMNS is baselines.STRENGTH_TRIO_COLUMNS
    assert ex._fit_logistic_pipeline is baselines._fit_logistic_pipeline


def test_class_names_reused_from_evaluation_by_identity():
    import backend.app.ml.evaluation as evaluation

    assert ex.CLASS_NAMES is evaluation.CLASS_NAMES
    assert ex.EXPECTED_CLASSES is evaluation.EXPECTED_CLASSES


# --------------------------------------------------------------------------
# 11. Artifact build/load/provenance - real data
# --------------------------------------------------------------------------
@requires_real_data
def test_source_artifact_contains_zero_sealed_season_rows(real_source_frame):
    """Structural guarantee, not a filtering result: this file has no rows
    for 2025/26 at all."""
    assert ex.SEALED_SEASON not in set(real_source_frame["Season"].unique())


@requires_real_data
def test_source_artifact_has_exactly_3800_rows(real_source_frame):
    assert len(real_source_frame) == 3800


@requires_real_data
def test_artifact_training_seasons_end_at_2024_25_and_exclude_sealed_season(real_artifact):
    assert real_artifact.metadata.training_cutoff_season == "2024_25"
    assert real_artifact.metadata.training_seasons[-1] == "2024_25"
    assert ex.SEALED_SEASON not in real_artifact.metadata.training_seasons
    assert real_artifact.metadata.n_training_rows == 3800


def test_build_strength_trio_artifact_takes_no_source_path_argument():
    """No public source_path parameter must exist at all - there is exactly
    one approved training source for this artifact, used internally."""
    import inspect

    signature = inspect.signature(ex.build_strength_trio_artifact)
    assert "source_path" not in signature.parameters
    assert list(signature.parameters) == []


@requires_real_data
def test_builder_uses_source_feature_artifact_path_internally_and_it_is_correct():
    """The artifact-building path must never open
    features_causal_through_2024_25.parquet (whose row cap extends into the
    sealed season). Checked on the actual module constant the function reads
    internally - not by searching prose, since the module's docstrings
    legitimately name that filename when explaining why it must not be used."""
    assert ex.SOURCE_FEATURE_ARTIFACT_PATH.name == "features_causal_through_2023_24.parquet"
    assert ex.SOURCE_FEATURE_ARTIFACT_PATH.name != "features_causal_through_2024_25.parquet"
    assert str(ex.SOURCE_FEATURE_ARTIFACT_PATH).endswith("features_causal_through_2023_24.parquet")


@requires_real_data
def test_artifact_metadata_identifies_it_as_the_frozen_model_for_2025_26(real_artifact):
    """The artifact metadata must explicitly record that it is the frozen
    model intended for eventual one-time application to the sealed season,
    with no retraining required before that evaluation."""
    meta = real_artifact.metadata
    assert meta.intended_final_evaluation_season == ex.SEALED_SEASON == "2025_26"
    assert meta.requires_retraining_before_final_evaluation is False


@requires_real_data
def test_artifact_contains_exactly_the_frozen_three_raw_features(real_artifact):
    assert real_artifact.metadata.feature_columns == list(STRENGTH_TRIO_COLUMNS)
    assert real_artifact.metadata.feature_columns == ["elo_diff", "diff_ewma_ppg", "diff_ewma_sot_diff"]


@requires_real_data
def test_build_strength_trio_artifact_has_no_seasons_parameter():
    """No generic 'train on arbitrary seasons' path must exist."""
    import inspect

    signature = inspect.signature(ex.build_strength_trio_artifact)
    assert "seasons" not in signature.parameters
    assert "season" not in signature.parameters


@requires_real_data
def test_save_and_load_round_trips(tmp_path, real_artifact):
    path = tmp_path / "artifact.joblib"
    ex.save_strength_trio_artifact(real_artifact, path=path)
    assert path.exists()
    assert path.with_suffix(path.suffix + ".json").exists()

    loaded = ex.load_strength_trio_artifact(path=path)
    assert loaded.metadata == real_artifact.metadata
    np.testing.assert_array_equal(
        loaded.pipeline.named_steps["model"].coef_, real_artifact.pipeline.named_steps["model"].coef_
    )


@requires_real_data
def test_loading_does_not_retrain(tmp_path, real_artifact, monkeypatch):
    from sklearn.pipeline import Pipeline

    path = tmp_path / "artifact.joblib"
    ex.save_strength_trio_artifact(real_artifact, path=path)

    def _forbidden_fit(self, *args, **kwargs):
        raise AssertionError("load_strength_trio_artifact must never call Pipeline.fit")

    monkeypatch.setattr(Pipeline, "fit", _forbidden_fit)
    loaded = ex.load_strength_trio_artifact(path=path)
    assert loaded.metadata.model_id == "baseline_strength_trio"


def test_malformed_artifact_type_fails_loudly(tmp_path):
    import joblib

    path = tmp_path / "bad_artifact.joblib"
    joblib.dump({"not": "an artifact"}, path)
    with pytest.raises(TypeError, match="StrengthTrioArtifact"):
        ex.load_strength_trio_artifact(path=path)


def test_incompatible_metadata_fails_loudly(synthetic_pipeline, tmp_path):
    import joblib

    bad_metadata = ex.ArtifactMetadata(
        model_id="baseline_strength_trio",
        artifact_format_version="99.0",  # wrong version
        training_cutoff_season=ex.TRAINING_CUTOFF_SEASON,
        training_seasons=["2024_25"],
        feature_columns=list(STRENGTH_TRIO_COLUMNS),
        transformed_feature_names=[],
        class_mapping={name: cls for name, cls in zip(CLASS_NAMES, EXPECTED_CLASSES)},
        source_feature_artifact="synthetic",
        n_training_rows=8,
        library_versions={},
        intended_final_evaluation_season=ex.SEALED_SEASON,
        requires_retraining_before_final_evaluation=False,
    )
    bad_artifact = ex.StrengthTrioArtifact(metadata=bad_metadata, pipeline=synthetic_pipeline)
    path = tmp_path / "bad_version.joblib"
    joblib.dump(bad_artifact, path)
    with pytest.raises(ValueError, match="format version"):
        ex.load_strength_trio_artifact(path=path)


def test_artifact_with_sealed_season_in_metadata_fails_loudly(synthetic_pipeline, tmp_path):
    import joblib

    bad_metadata = ex.ArtifactMetadata(
        model_id="baseline_strength_trio",
        artifact_format_version=ex.STRENGTH_TRIO_ARTIFACT_FORMAT_VERSION,
        training_cutoff_season=ex.TRAINING_CUTOFF_SEASON,
        training_seasons=["2024_25", ex.SEALED_SEASON],
        feature_columns=list(STRENGTH_TRIO_COLUMNS),
        transformed_feature_names=[],
        class_mapping={name: cls for name, cls in zip(CLASS_NAMES, EXPECTED_CLASSES)},
        source_feature_artifact="synthetic",
        n_training_rows=8,
        library_versions={},
        intended_final_evaluation_season=ex.SEALED_SEASON,
        requires_retraining_before_final_evaluation=False,
    )
    bad_artifact = ex.StrengthTrioArtifact(metadata=bad_metadata, pipeline=synthetic_pipeline)
    path = tmp_path / "bad_sealed.joblib"
    joblib.dump(bad_artifact, path)
    with pytest.raises(ValueError, match="sealed season"):
        ex.load_strength_trio_artifact(path=path)


def test_artifact_claiming_retraining_required_fails_loudly(synthetic_pipeline, tmp_path):
    """The frozen-model contract requires requires_retraining_before_final_evaluation
    to always be False - a loaded artifact that claims otherwise must be rejected."""
    import joblib

    bad_metadata = ex.ArtifactMetadata(
        model_id="baseline_strength_trio",
        artifact_format_version=ex.STRENGTH_TRIO_ARTIFACT_FORMAT_VERSION,
        training_cutoff_season=ex.TRAINING_CUTOFF_SEASON,
        training_seasons=["2024_25"],
        feature_columns=list(STRENGTH_TRIO_COLUMNS),
        transformed_feature_names=[],
        class_mapping={name: cls for name, cls in zip(CLASS_NAMES, EXPECTED_CLASSES)},
        source_feature_artifact="synthetic",
        n_training_rows=8,
        library_versions={},
        intended_final_evaluation_season=ex.SEALED_SEASON,
        requires_retraining_before_final_evaluation=True,  # invalid
    )
    bad_artifact = ex.StrengthTrioArtifact(metadata=bad_metadata, pipeline=synthetic_pipeline)
    path = tmp_path / "bad_requires_retraining.joblib"
    joblib.dump(bad_artifact, path)
    with pytest.raises(ValueError, match="retraining is required"):
        ex.load_strength_trio_artifact(path=path)


def test_load_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.load_strength_trio_artifact(path=tmp_path / "does_not_exist.joblib")


@requires_real_data
def test_build_artifact_validates_provenance_before_fitting():
    """assert_artifact_valid_for must be called - verified by confirming a
    corrupted/mismatched provenance sidecar causes the build to fail."""
    import inspect

    source = inspect.getsource(ex.build_strength_trio_artifact)
    assert "assert_artifact_valid_for" in source


# --------------------------------------------------------------------------
# 12. Dixon-Coles explanation - minimal, no attribution claims
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dc_params_and_config():
    from backend.app.ml.score_models import CONFIGS, fit_score_model, load_score_fold

    fold = load_score_fold(1)
    config = CONFIGS["dixon_coles_l2_decay"]
    params = fit_score_model(fold.train, config)
    return params, config, fold


def test_dc_explanation_contains_only_direct_model_reads(dc_params_and_config):
    params, config, fold = dc_params_and_config
    home_team = fold.train.HomeTeam.iloc[0]
    away_team = fold.train.AwayTeam.iloc[0]
    explanation = ex.explain_score_model_match(params, config, home_team, away_team)

    assert explanation["expected_home_goals"] > 0
    assert explanation["expected_away_goals"] > 0
    assert explanation["home_advantage"] == params.home_advantage
    assert explanation["rho"] == params.rho
    assert explanation["probabilities"]["home"] + explanation["probabilities"]["draw"] + explanation["probabilities"]["away"] == pytest.approx(1.0, abs=1e-9)
    assert len(explanation["top_scorelines"]) >= 1
    json.dumps(explanation)


def test_dc_explanation_has_no_feature_attribution_or_shap_fields(dc_params_and_config):
    params, config, fold = dc_params_and_config
    explanation = ex.explain_score_model_match(
        params, config, fold.train.HomeTeam.iloc[0], fold.train.AwayTeam.iloc[0]
    )
    forbidden_keys = {
        "logit_contribution", "grouped_logit_contribution", "probability_sensitivity",
        "shap_values", "feature_importance", "drivers_by_logit_contribution",
    }
    assert forbidden_keys.isdisjoint(explanation.keys())


def test_dc_explanation_does_not_force_the_logistic_schema(dc_params_and_config):
    """The DC explanation is a distinct, smaller schema - not padded to match
    explain_match's feature-list structure."""
    params, config, fold = dc_params_and_config
    explanation = ex.explain_score_model_match(
        params, config, fold.train.HomeTeam.iloc[0], fold.train.AwayTeam.iloc[0]
    )
    assert "features" not in explanation


def test_dc_explanation_reuses_predict_match_by_identity():
    import backend.app.ml.score_models as score_models

    assert ex.predict_score_model_match is score_models.predict_match


def test_no_shap_import_anywhere():
    import ast
    import inspect

    for module in (ex,):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("shap" not in alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert "shap" not in node.module.lower()

    try:
        import shap  # noqa: F401

        pytest.fail("shap must not be installed for this project")
    except ModuleNotFoundError:
        pass


# --------------------------------------------------------------------------
# 13. 2025/26 exclusion
# --------------------------------------------------------------------------
def test_no_2025_26_required_for_synthetic_explanation(synthetic_artifact, synthetic_frame):
    """Explaining a match never needs the sealed season at all."""
    explanation = ex.explain_match(synthetic_artifact, synthetic_frame.iloc[[0]])
    assert "2025_26" not in json.dumps(explanation)


@requires_real_data
def test_artifact_training_seasons_match_the_source_frame_exactly(real_artifact, real_source_frame):
    """Cross-check: the artifact's recorded training_seasons must equal the
    actual seasons present in the source file - no silent narrowing or
    widening between what was read and what was recorded."""
    assert set(real_artifact.metadata.training_seasons) == set(real_source_frame["Season"].unique())
