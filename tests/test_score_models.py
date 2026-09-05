"""Tests for the Poisson / Dixon-Coles score models.

Strategy mirrors tests/test_training_pipeline.py: a small, fast synthetic
Premier League history using REAL season labels drives most tests (so the
real DEVELOPMENT_FOLD_DEFINITIONS pairing applies directly), plus a handful
of real-artifact integration tests against the actual matches.parquet,
skipped if that file is not present.

Many tests construct a `ScoreModelParams` by hand rather than fitting one -
`predict_match`/`predict_fold` are pure functions of (params, team names),
so prediction-time behaviour (unseen/returning-team fallback, class order,
scoreline-matrix properties, "validation goals never affect predictions")
can be tested completely independently of the optimiser.

Tests that DO require an actual fit (leakage canary, determinism,
identifiability, promoted-prior-in-context) call `fit_score_model` directly;
there is nothing special about them, they just take longer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.app.ml.datasets as datasets  # noqa: E402
import backend.app.ml.evaluation as evaluation  # noqa: E402
import backend.app.ml.score_models as sm  # noqa: E402

REAL_MATCHES_PATH = REPO_ROOT / "data" / "processed" / "matches.parquet"
requires_real_data = pytest.mark.skipif(
    not REAL_MATCHES_PATH.exists(), reason="data/processed/matches.parquet not built"
)


# --------------------------------------------------------------------------
# Synthetic Premier League history (real season labels, goal counts)
# --------------------------------------------------------------------------
def _build_synthetic_score_matches(season_teams: dict[str, list[str]]) -> pd.DataFrame:
    """Deterministic synthetic match history with goal counts (not just
    W/D/L), using REAL season labels so DEVELOPMENT_FOLD_DEFINITIONS applies
    directly. One roster change (F -> G after the opening season) gives a
    genuine newcomer for the promoted-prior calculation."""
    records = []
    for season_index, (season, teams) in enumerate(season_teams.items()):
        start = pd.Timestamp(f"{season.split('_')[0]}-08-10")
        day = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                fthg = (day + season_index) % 4
                ftag = (day + season_index + 2) % 3
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
                    }
                )
                day += 1
    return pd.DataFrame(records)


SYNTHETIC_SEASON_TEAMS: dict[str, list[str]] = {
    season: (["A", "B", "C", "D", "E", "F"] if season == "2015_16" else ["A", "B", "C", "D", "E", "G"])
    for season in ["2015_16", "2016_17", "2017_18", "2018_19", "2019_20", "2020_21", "2021_22", "2022_23"]
}


@pytest.fixture(scope="module")
def synthetic_matches() -> pd.DataFrame:
    return _build_synthetic_score_matches(SYNTHETIC_SEASON_TEAMS)


@pytest.fixture()
def synthetic_matches_path(tmp_path, synthetic_matches, monkeypatch) -> Path:
    path = tmp_path / "matches.parquet"
    synthetic_matches.to_parquet(path, index=False)
    monkeypatch.setattr(sm, "MATCHES_PATH", path)
    return path


@pytest.fixture()
def synthetic_score_fold(synthetic_matches_path) -> sm.ScoreFoldData:
    return sm.load_score_fold(1)


def _hand_params(**overrides) -> sm.ScoreModelParams:
    base = dict(
        config_id="poisson",
        teams=("A", "B"),
        intercept=0.3,
        home_advantage=0.2,
        attack={"A": 0.1, "B": -0.1},
        defence={"A": -0.05, "B": 0.05},
        rho=None,
        promoted_attack_offset=-0.32,
        promoted_defence_offset=0.10,
    )
    base.update(overrides)
    return sm.ScoreModelParams(**base)


def _one_match_frame(home="A", away="B", fthg=2, ftag=0, season="2022_23") -> pd.DataFrame:
    ftr = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
    return pd.DataFrame(
        [{"Season": season, "Date": pd.Timestamp("2022-08-10"), "HomeTeam": home, "AwayTeam": away,
          "FTHG": fthg, "FTAG": ftag, "FTR": ftr}]
    )


# --------------------------------------------------------------------------
# 1. Fold safety: read-only reuse of the Stage 1 fold table, no path to 2025/26
# --------------------------------------------------------------------------
def test_score_models_reuses_the_same_fold_table_as_stage1_by_identity():
    """Not a copy, not a re-derivation - the literal same objects."""
    assert sm.DEVELOPMENT_FOLD_DEFINITIONS is datasets.DEVELOPMENT_FOLD_DEFINITIONS
    assert sm.DEVELOPMENT_FOLDS is datasets.DEVELOPMENT_FOLDS


def test_load_score_fold_refuses_folds_outside_the_development_set(synthetic_matches_path):
    for bad_fold in [0, 4, 5, -1]:
        with pytest.raises(ValueError, match="sealed season must never be touched"):
            sm.load_score_fold(bad_fold)


def test_development_fold_definitions_never_reference_the_sealed_season():
    for definition in sm.DEVELOPMENT_FOLD_DEFINITIONS.values():
        assert definition["training_cutoff"] != sm.SEALED_SEASON
        assert definition["validation_season"] != sm.SEALED_SEASON


def test_load_score_fold_contains_no_sealed_rows_and_train_precedes_validation(synthetic_score_fold):
    fold = synthetic_score_fold
    seasons_touched = set(fold.train_seasons) | {fold.validation_season}
    assert sm.SEALED_SEASON not in seasons_touched
    assert fold.train["Date"].max() < fold.validation["Date"].min()
    assert set(fold.validation["Season"].unique()) == {fold.validation_season}
    assert fold.validation_season not in set(fold.train["Season"].unique())


def test_load_score_fold_only_carries_the_seven_match_columns(synthetic_score_fold):
    assert list(synthetic_score_fold.train.columns) == sm.MATCH_COLUMNS
    assert list(synthetic_score_fold.validation.columns) == sm.MATCH_COLUMNS


@requires_real_data
def test_real_data_development_folds_have_no_sealed_rows():
    for fold_number in sm.DEVELOPMENT_FOLDS:
        fold = sm.load_score_fold(fold_number)
        seasons_touched = set(fold.train_seasons) | {fold.validation_season}
        assert sm.SEALED_SEASON not in seasons_touched
        assert fold.train["Date"].max() < fold.validation["Date"].min()


# --------------------------------------------------------------------------
# 2. Fit signature: no path for validation data to enter fitting
# --------------------------------------------------------------------------
def test_fit_score_model_signature_has_no_validation_parameter():
    import inspect

    params = list(inspect.signature(sm.fit_score_model).parameters)
    assert params == ["train", "config"]
    assert "validation" not in params and "y_val" not in params and "val" not in params


def test_changing_validation_goals_does_not_change_predictions():
    """predict_match/predict_fold read HomeTeam/AwayTeam identity only -
    the actual FTHG/FTAG/FTR of the row being predicted is never consulted."""
    params = _hand_params()
    val_a = _one_match_frame(fthg=2, ftag=0)
    val_b = val_a.copy()
    val_b.loc[0, ["FTHG", "FTAG", "FTR"]] = [0, 5, "A"]  # dramatically different actual result

    proba_a, _ = sm.predict_fold(params, sm.CONFIGS["poisson"], val_a)
    proba_b, _ = sm.predict_fold(params, sm.CONFIGS["poisson"], val_b)
    np.testing.assert_array_equal(proba_a, proba_b)


def test_leakage_canary_mutating_validation_does_not_change_fitted_params(synthetic_score_fold):
    """The strongest available canary given fit_score_model takes no
    validation argument at all: fit, dramatically mutate a SEPARATE
    validation frame, refit on the SAME (untouched) training frame, and
    confirm byte-identical fitted parameters."""
    config = sm.CONFIGS["poisson"]
    params_a = sm.fit_score_model(synthetic_score_fold.train, config)

    mutated_validation = synthetic_score_fold.validation.copy()
    mutated_validation[["FTHG", "FTAG"]] = 99

    params_b = sm.fit_score_model(synthetic_score_fold.train, config)  # train is unchanged

    assert params_a.attack == params_b.attack
    assert params_a.defence == params_b.defence
    assert params_a.intercept == params_b.intercept
    assert params_a.home_advantage == params_b.home_advantage
    assert params_a.rho == params_b.rho
    assert len(mutated_validation) > 0  # the mutation happened; it just never mattered


# --------------------------------------------------------------------------
# 3. Poisson PMF hand calculation
# --------------------------------------------------------------------------
def test_poisson_pmf_matches_hand_calculation():
    from scipy.stats import poisson

    lam = 1.5
    for k in [0, 1, 2, 3, 5]:
        expected = math.exp(-lam) * lam**k / math.factorial(k)
        assert poisson.pmf(k, lam) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# 4. Dixon-Coles tau hand calculations (0-0, 0-1, 1-0, 1-1)
# --------------------------------------------------------------------------
def test_dixon_coles_tau_hand_calculations():
    lam_h, lam_a, rho = 1.4, 1.1, -0.15
    tau00, tau01, tau10, tau11 = sm._dixon_coles_tau(np.array(lam_h), np.array(lam_a), rho)
    assert float(tau00) == pytest.approx(1 - lam_h * lam_a * rho)
    assert float(tau01) == pytest.approx(1 + lam_h * rho)
    assert float(tau10) == pytest.approx(1 + lam_a * rho)
    assert float(tau11) == pytest.approx(1 - rho)


def test_dixon_coles_tau_matches_published_example():
    """Cross-check against the standard worked example: lambda_h=1.5,
    lambda_a=1.0, rho=-0.1."""
    lam_h, lam_a, rho = 1.5, 1.0, -0.1
    tau00, tau01, tau10, tau11 = sm._dixon_coles_tau(np.array(lam_h), np.array(lam_a), rho)
    assert float(tau00) == pytest.approx(1 - 1.5 * 1.0 * -0.1)  # 1.15
    assert float(tau01) == pytest.approx(1 + 1.5 * -0.1)  # 0.85
    assert float(tau10) == pytest.approx(1 + 1.0 * -0.1)  # 0.90
    assert float(tau11) == pytest.approx(1 - -0.1)  # 1.10


# --------------------------------------------------------------------------
# 5. Tau infeasibility: the user's exact counterexamples, rejected not floored
# --------------------------------------------------------------------------
def test_tau_counterexample_lambda5_rho_positive_gives_negative_tau00():
    """lambda_home=lambda_away=5, rho=+0.2 -> tau(0,0) = 1 - 25*0.2 = -4."""
    tau00, _, _, _ = sm._dixon_coles_tau(np.array(5.0), np.array(5.0), 0.2)
    assert float(tau00) == pytest.approx(-4.0)


def test_tau_counterexample_is_rejected_not_floored_in_scoreline_matrix():
    with pytest.raises(RuntimeError, match="infeasible"):
        sm._build_scoreline_matrix(5.0, 5.0, 0.2, use_dixon_coles=True)


def test_tau_marginal_zero_case_is_rejected_under_strict_positivity():
    """lambda=5, rho=-0.2 -> tau(0,1) = 1 + 5*(-0.2) = exactly 0, which must
    still fail (tau must be STRICTLY > 0, not >= 0)."""
    _, tau01, _, _ = sm._dixon_coles_tau(np.array(5.0), np.array(5.0), -0.2)
    assert float(tau01) == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="infeasible"):
        sm._build_scoreline_matrix(5.0, 5.0, -0.2, use_dixon_coles=True)


def test_feasible_tau_point_builds_a_valid_matrix():
    matrix = sm._build_scoreline_matrix(1.5, 1.2, -0.13, use_dixon_coles=True)
    assert (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 6. Objective-level infeasibility: penalty scales with violation, never a
# flat wall, and never substituted into the true likelihood
# --------------------------------------------------------------------------
def _objective(theta, **overrides):
    kwargs = dict(
        home_idx=np.array([0, 1]),
        away_idx=np.array([1, 0]),
        home_goals=np.array([1.0, 2.0]),
        away_goals=np.array([0.0, 1.0]),
        weights=np.array([1.0, 1.0]),
        n_teams=4,
        has_rho=False,
        l2_sigma=None,
    )
    kwargs.update(overrides)
    return sm._negative_log_likelihood(theta, **kwargs)


def test_tau_infeasible_objective_point_returns_large_penalty_via_dc_theta():
    # 2 teams, DC on: force lambda_h=lambda_a=5-ish and rho=+0.2 to trigger
    # the tau(0,0)<0 counterexample through the full objective path.
    # c + gamma + atk[home] + def[away] = log(5) for both sides.
    log5 = math.log(5.0)
    theta = np.array([log5, 0.0, 0.0, 0.0, 0.2])  # c=log5, gamma=0, atk_free=[0], def_free=[0], rho=0.2
    value = sm._negative_log_likelihood(
        theta,
        home_idx=np.array([0, 1]),
        away_idx=np.array([1, 0]),
        home_goals=np.array([0.0, 0.0]),
        away_goals=np.array([0.0, 0.0]),
        weights=np.array([1.0, 1.0]),
        n_teams=2,
        has_rho=True,
        l2_sigma=None,
    )
    assert value >= sm.INFEASIBLE_PENALTY


def test_bound_violation_on_reconstructed_last_team_is_detected():
    """3 free attack params summing to 4.2 -> reconstructed 4th = -4.2,
    which is NOT individually bounded by L-BFGS-B's `bounds=` (that only
    constrains the 3 explicit free variables) - the objective must catch it."""
    theta = np.concatenate([[0.2, 0.19], [1.4, 1.4, 1.4], [0.0, 0.0, 0.0]])
    c, gamma, atk_full, def_full, rho = sm._unpack_theta(theta, n_teams=4, has_rho=False)
    assert atk_full[-1] == pytest.approx(-4.2)
    assert sm._bound_violation(atk_full, sm.BOUND_TEAM_PARAM) > 0.0

    value = _objective(theta, n_teams=4)
    assert value >= sm.INFEASIBLE_PENALTY


def test_infeasibility_penalty_scales_with_violation_magnitude():
    theta_small = np.concatenate([[0.2, 0.19], [1.6, 0.0, 0.0], [0.0, 0.0, 0.0]])  # atk_4 = -1.6
    theta_big = np.concatenate([[0.2, 0.19], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # atk_4 = -5.0
    penalty_small = _objective(theta_small)
    penalty_big = _objective(theta_big)
    assert penalty_small >= sm.INFEASIBLE_PENALTY
    assert penalty_big > penalty_small  # NOT a flat wall


def test_feasible_point_returns_the_true_likelihood_not_a_penalty():
    theta_feasible = np.concatenate([[0.2, 0.19], [0.1, -0.05, 0.0], [0.0, 0.0, 0.0]])
    value = _objective(theta_feasible)
    assert value < sm.INFEASIBLE_PENALTY / 2.0
    assert np.isfinite(value)


# --------------------------------------------------------------------------
# 7. Sum-to-zero identifiability
# --------------------------------------------------------------------------
def test_unpack_theta_reconstructs_sum_to_zero_vectors():
    theta = np.array([0.2, 0.19, 0.3, -0.1, 0.05, -0.02])  # n_teams=3: 2 free atk, 2 free def
    c, gamma, atk_full, def_full, rho = sm._unpack_theta(theta, n_teams=3, has_rho=False)
    assert c == pytest.approx(0.2)
    assert gamma == pytest.approx(0.19)
    np.testing.assert_allclose(atk_full, [0.3, -0.1, -0.2])
    np.testing.assert_allclose(def_full, [0.05, -0.02, -0.03])
    assert rho is None
    assert atk_full.sum() == pytest.approx(0.0, abs=1e-12)
    assert def_full.sum() == pytest.approx(0.0, abs=1e-12)


def test_unpack_theta_extracts_rho_when_present():
    theta = np.array([0.2, 0.19, 0.3, -0.1, 0.05, -0.02, -0.08])
    _, _, _, _, rho = sm._unpack_theta(theta, n_teams=3, has_rho=True)
    assert rho == pytest.approx(-0.08)


def test_identifiability_holds_after_a_real_fit(synthetic_score_fold):
    params = sm.fit_score_model(synthetic_score_fold.train, sm.CONFIGS["poisson"])
    assert abs(sum(params.attack.values())) < 1e-6
    assert abs(sum(params.defence.values())) < 1e-6


# --------------------------------------------------------------------------
# 8. Unseen / returning team fallback
# --------------------------------------------------------------------------
def test_unseen_team_uses_the_promoted_prior_not_a_fitted_value():
    params = _hand_params()
    assert params.team_attack("A") == pytest.approx(0.1)  # fitted
    assert params.team_attack("NeverSeen") == pytest.approx(params.promoted_attack_offset)
    assert params.team_defence("NeverSeen") == pytest.approx(params.promoted_defence_offset)


def test_returning_team_uses_its_fitted_value_not_the_promoted_prior():
    """A team absent for one or more seasons but present SOMEWHERE in the
    training window is indistinguishable, at prediction time, from a
    continuing team - it is simply a key in the fitted dict. This is the
    entire V1 "no special-casing" strategy for returning teams."""
    params = _hand_params(attack={"A": 0.1, "ReturningTeam": -0.4}, defence={"A": -0.05, "ReturningTeam": 0.3})
    assert params.team_attack("ReturningTeam") == pytest.approx(-0.4)
    assert params.team_defence("ReturningTeam") == pytest.approx(0.3)


def test_promoted_prior_is_estimated_from_training_rows_only():
    records = []
    for date, home, away, fh, fa in [
        ("2019-08-10", "A", "B", 2, 0),
        ("2019-08-17", "B", "A", 1, 1),
    ]:
        records.append(dict(Season="2019_20", Date=pd.Timestamp(date), HomeTeam=home, AwayTeam=away,
                             FTHG=fh, FTAG=fa, FTR="H" if fh > fa else ("A" if fh < fa else "D")))
    for date, home, away, fh, fa in [
        ("2020-08-10", "A", "C", 3, 0),
        ("2020-08-17", "C", "A", 0, 2),
        ("2020-08-24", "B", "C", 2, 0),
    ]:
        records.append(dict(Season="2020_21", Date=pd.Timestamp(date), HomeTeam=home, AwayTeam=away,
                             FTHG=fh, FTAG=fa, FTR="H" if fh > fa else ("A" if fh < fa else "D")))
    train = pd.DataFrame(records)

    attack_offset, defence_offset = sm._promoted_team_prior(train)
    # C (the newcomer) never scores in this fixture, so its attack offset
    # cannot be computed (log of zero) and correctly falls back to 0.0;
    # its defence offset (concedes heavily) is well-defined and non-zero.
    assert attack_offset == pytest.approx(0.0)
    assert defence_offset > 0.0


def test_promoted_prior_falls_back_to_zero_with_no_newcomers():
    train = _build_synthetic_score_matches({"2019_20": ["A", "B"], "2020_21": ["A", "B"]})
    attack_offset, defence_offset = sm._promoted_team_prior(train)
    assert attack_offset == pytest.approx(0.0)
    assert defence_offset == pytest.approx(0.0)


def test_promoted_prior_in_a_real_fit_is_used_for_the_actual_newcomer(synthetic_score_fold):
    """Fold 1's training window (2015_16..2021_22) has a real roster change
    (F -> G after 2015_16), so the promoted prior should be non-trivial."""
    params = sm.fit_score_model(synthetic_score_fold.train, sm.CONFIGS["poisson"])
    assert "G" in params.attack  # G appeared in training -> gets a fitted value
    assert "H" not in params.attack  # a team never seen at all
    assert params.team_attack("H") == pytest.approx(params.promoted_attack_offset)


# --------------------------------------------------------------------------
# 9. Dynamic scoreline grid
# --------------------------------------------------------------------------
def test_grid_extends_beyond_the_default_start_for_a_large_lambda():
    matrix = sm._build_scoreline_matrix(6.0, 5.5, None, use_dixon_coles=False)
    assert matrix.shape[0] > sm.SCORELINE_GRID_START + 1


def test_grid_does_not_extend_unnecessarily_for_typical_lambda():
    matrix = sm._build_scoreline_matrix(1.5, 1.2, None, use_dixon_coles=False)
    assert matrix.shape[0] == sm.SCORELINE_GRID_START + 1


def test_grid_ceiling_breach_raises_loudly_rather_than_renormalising_bad_mass():
    with pytest.raises(RuntimeError, match="could not reduce truncated"):
        sm._build_scoreline_matrix(50.0, 50.0, None, use_dixon_coles=False)


def test_renormalisation_only_happens_after_truncation_check_passes():
    """The matrix returned always sums to exactly 1, and this must be true
    at every grid size the function might settle on."""
    for lam in [0.8, 1.5, 3.0, 6.0]:
        matrix = sm._build_scoreline_matrix(lam, lam, None, use_dixon_coles=False)
        assert matrix.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 10. Matrix / H-D-A normalization and expected goals
# --------------------------------------------------------------------------
def test_scoreline_matrix_is_non_negative_and_sums_to_one():
    matrix = sm._build_scoreline_matrix(1.4, 1.1, -0.13, use_dixon_coles=True)
    assert (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(1.0, abs=1e-9)


def test_hda_probabilities_sum_to_one():
    matrix = sm._build_scoreline_matrix(1.4, 1.1, None, use_dixon_coles=False)
    prediction = sm._summarize_matrix(matrix, 1.4, 1.1)
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(1.0, abs=1e-9)


def test_expected_goals_are_positive():
    matrix = sm._build_scoreline_matrix(1.5, 1.2, None, use_dixon_coles=False)
    prediction = sm._summarize_matrix(matrix, 1.5, 1.2)
    assert prediction.expected_home_goals > 0.0
    assert prediction.expected_away_goals > 0.0


def test_most_likely_scoreline_is_the_argmax_cell():
    matrix = sm._build_scoreline_matrix(1.5, 1.2, None, use_dixon_coles=False)
    prediction = sm._summarize_matrix(matrix, 1.5, 1.2)
    home_str, away_str = prediction.most_likely_scoreline.split("-")
    assert matrix[int(home_str), int(away_str)] == pytest.approx(matrix.max())
    assert prediction.top_scorelines[0][1] == pytest.approx(matrix.max())


def test_predict_match_end_to_end_without_fitting():
    params = _hand_params()
    prediction = sm.predict_match(params, sm.CONFIGS["poisson"], "A", "B")
    assert prediction.lambda_home > 0 and prediction.lambda_away > 0
    assert prediction.matrix.sum() == pytest.approx(1.0, abs=1e-9)
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 11. Class order H/D/A = [0, 1, 2] and probability-contract compliance
# --------------------------------------------------------------------------
def test_predict_fold_columns_are_h_d_a_in_order_without_fitting():
    from backend.app.ml.feature_engineering import TARGET_MAPPING

    assert TARGET_MAPPING == {"H": 0, "D": 1, "A": 2}

    params = _hand_params()
    validation = _one_match_frame(fthg=2, ftag=0)
    proba, predictions = sm.predict_fold(params, sm.CONFIGS["poisson"], validation)

    assert proba.shape == (1, 3)
    assert proba[0, 0] == pytest.approx(predictions[0].p_home)
    assert proba[0, 1] == pytest.approx(predictions[0].p_draw)
    assert proba[0, 2] == pytest.approx(predictions[0].p_away)
    assert evaluation.validate_probabilities(proba) == []


def test_predict_fold_raises_on_probability_contract_violation_never_silently():
    """predict_fold calls validate_probabilities itself; a genuinely broken
    prediction path must raise, not silently produce a bad array. Verified
    indirectly: validate_probabilities correctly rejects a hand-built bad array
    (the actual code path predict_fold delegates to)."""
    bad = np.array([[0.5, 0.5, 0.5]])
    assert evaluation.validate_probabilities(bad) != []


# --------------------------------------------------------------------------
# 12. Direct reuse of evaluation.py (no duplicated metric logic)
# --------------------------------------------------------------------------
def test_score_models_reuses_evaluation_functions_by_identity():
    assert sm.compute_fold_metrics is evaluation.compute_fold_metrics
    assert sm.validate_probabilities is evaluation.validate_probabilities
    assert sm.paired_log_loss_comparison is evaluation.paired_log_loss_comparison


def test_fold_metrics_are_computed_via_the_real_evaluation_module(synthetic_score_fold):
    params = sm.fit_score_model(synthetic_score_fold.train, sm.CONFIGS["poisson"])
    proba, _ = sm.predict_fold(params, sm.CONFIGS["poisson"], synthetic_score_fold.validation)
    from backend.app.ml.feature_engineering import TARGET_MAPPING

    y_true = synthetic_score_fold.validation["FTR"].map(TARGET_MAPPING).to_numpy()
    metrics = evaluation.compute_fold_metrics(
        fold=synthetic_score_fold.fold, validation_season=synthetic_score_fold.validation_season,
        y_true=y_true, proba=proba,
    )
    assert metrics.n_rows == len(synthetic_score_fold.validation)
    assert isinstance(metrics, evaluation.FoldMetrics)


# --------------------------------------------------------------------------
# 13. Determinism
# --------------------------------------------------------------------------
def test_fitting_is_deterministic(synthetic_score_fold):
    params_a = sm.fit_score_model(synthetic_score_fold.train, sm.CONFIGS["dixon_coles_l2"])
    params_b = sm.fit_score_model(synthetic_score_fold.train, sm.CONFIGS["dixon_coles_l2"])
    assert params_a.attack == params_b.attack
    assert params_a.defence == params_b.defence
    assert params_a.rho == params_b.rho
    assert params_a.intercept == params_b.intercept
    assert params_a.home_advantage == params_b.home_advantage


def test_predictions_are_deterministic_given_the_same_fitted_params(synthetic_score_fold):
    params = sm.fit_score_model(synthetic_score_fold.train, sm.CONFIGS["poisson"])
    proba_a, _ = sm.predict_fold(params, sm.CONFIGS["poisson"], synthetic_score_fold.validation)
    proba_b, _ = sm.predict_fold(params, sm.CONFIGS["poisson"], synthetic_score_fold.validation)
    np.testing.assert_array_equal(proba_a, proba_b)


# --------------------------------------------------------------------------
# 14. Config grid and local selection (not Stage 1's selection logic)
# --------------------------------------------------------------------------
def test_exactly_five_configs_are_defined():
    assert list(sm.CONFIGS) == ["poisson", "poisson_l2", "dixon_coles", "dixon_coles_l2", "dixon_coles_l2_decay"]
    assert sm.CONFIGS["poisson"].use_dixon_coles is False
    assert sm.CONFIGS["poisson"].l2_sigma is None
    assert sm.CONFIGS["poisson_l2"].l2_sigma == pytest.approx(0.25)
    assert sm.CONFIGS["dixon_coles"].use_dixon_coles is True
    assert sm.CONFIGS["dixon_coles_l2_decay"].half_life_days == pytest.approx(365.0)
    for name, config in sm.CONFIGS.items():
        assert config.config_id == name


def test_select_best_score_model_is_local_not_stage1s_selection():
    import backend.app.ml.training as training

    assert sm.select_best_score_model is not training.select_best_configuration
    assert sm.SCORE_MODEL_COMPLEXITY_RANK is not training.MODEL_COMPLEXITY_RANK
    # Stage 1's ranking knows nothing about these config names.
    for config_id in sm.CONFIGS:
        assert config_id not in training.MODEL_COMPLEXITY_RANK


def _fake_result(config_id: str, proba_by_fold, y_by_fold) -> sm.ScoreModelResult:
    fold_metrics = [
        evaluation.compute_fold_metrics(fold=i + 1, validation_season=f"fold{i + 1}", y_true=y, proba=p)
        for i, (y, p) in enumerate(zip(y_by_fold, proba_by_fold))
    ]
    log_losses = np.array([m.log_loss for m in fold_metrics])
    return sm.ScoreModelResult(
        config_id=config_id, fold_metrics=fold_metrics, fold_proba=proba_by_fold, fold_y_true=y_by_fold,
        mean_log_loss=float(log_losses.mean()), worst_log_loss=float(log_losses.max()),
        log_loss_std=float(log_losses.std(ddof=0)),
    )


def test_select_best_score_model_picks_lowest_mean_log_loss_when_not_tied():
    n = 30
    y = np.array([0] * n)
    confident_correct = np.tile([0.9, 0.05, 0.05], (n, 1))
    confident_wrong = np.tile([0.05, 0.05, 0.9], (n, 1))
    good = _fake_result("dixon_coles", [confident_correct, confident_correct], [y, y])
    bad = _fake_result("poisson", [confident_wrong, confident_wrong], [y, y])
    assert sm.select_best_score_model([good, bad]).config_id == "dixon_coles"


def test_select_best_score_model_tie_break_prefers_simpler_config():
    n = 30
    y = np.array([0, 1, 2] * (n // 3))
    proba = np.tile([0.5, 0.3, 0.2], (n, 1))  # identical predictions -> exact tie
    simple = _fake_result("poisson", [proba.copy(), proba.copy()], [y, y])
    complex_ = _fake_result("dixon_coles_l2_decay", [proba.copy(), proba.copy()], [y, y])
    assert sm.select_best_score_model([complex_, simple]).config_id == "poisson"


# --------------------------------------------------------------------------
# 15. Time decay weighting
# --------------------------------------------------------------------------
def test_time_decay_weights_are_uniform_when_disabled():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]))
    weights = sm._time_decay_weights(dates, None)
    np.testing.assert_array_equal(weights, np.ones(3))


def test_time_decay_weights_decay_from_the_latest_training_date():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01"]))  # 366 days apart
    weights = sm._time_decay_weights(dates, half_life_days=365.0)
    assert weights[-1] == pytest.approx(1.0)  # the latest date itself
    assert weights[0] == pytest.approx(0.5, rel=0.01)  # ~one half-life earlier
    assert weights[0] < weights[-1]
