"""Development-only CLI for the calibration / ensemble meta-layer.

Runs the four pre-committed meta-candidates through strict chronological
DEVELOPMENT meta-validation (M1: fit 2022_23 -> evaluate 2023_24;
M2: fit 2022_23+2023_24 -> evaluate 2024_25), applies the conservative
selection rule, and writes reports under reports/modeling/.

This is development meta-validation, not a pristine final test: the design pass
that produced this stage already inspected development labels. The genuinely
sealed final test remains 2025/26, which this script can never reach - it reads
only the existing development OOF prediction CSVs, and `load_meta_predictions`
raises if a sealed-season row appears anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.calibration import (
    CANDIDATES,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_ROWS,
    HISTORICAL_TRIO_THREE_SEASON_LOG_LOSS,
    INCUMBENT_CANDIDATE,
    META_FOLD_DEFINITIONS,
    META_FOLDS,
    fit_final_meta_parameters,
    library_versions,
    load_meta_predictions,
    per_class_reliability,
    run_meta_validation,
    select_best_candidate,
)
from backend.app.ml.feature_engineering import SEALED_SEASON

REPORTS_DIR = REPO_ROOT / "reports" / "modeling"


def main(argv: list[str] | None = None) -> int:
    print("Loading frozen development OOF predictions ...")
    predictions = load_meta_predictions()
    print(f"  merged rows: {len(predictions)} (expected {EXPECTED_ROWS})")
    print(f"  seasons: {predictions.seasons}")

    print("\nStrict chronological meta-folds (DEVELOPMENT meta-validation):")
    for meta_fold in META_FOLDS:
        definition = META_FOLD_DEFINITIONS[meta_fold]
        train_rows = len(predictions.subset(definition["meta_train_seasons"]))
        eval_rows = len(predictions.subset([definition["evaluation_season"]]))
        print(
            f"  M{meta_fold}: meta-train {'+'.join(definition['meta_train_seasons'])} "
            f"({train_rows} rows) -> evaluate {definition['evaluation_season']} ({eval_rows} rows)"
        )

    results = []
    print(f"\nRunning {len(CANDIDATES)} pre-committed candidate(s) ...")
    for candidate in CANDIDATES.values():
        result = run_meta_validation(candidate, predictions)
        results.append(result)
        params = {
            f"M{r.meta_fold}": {k: round(v, 4) for k, v in r.fitted_params.items()}
            for r in result.fold_results
        }
        print(
            f"  {result.name:22s} mean={result.mean_log_loss:.4f}  "
            f"worst={result.worst_log_loss:.4f}  params={params}"
        )

    incumbent = next(r for r in results if r.name == INCUMBENT_CANDIDATE)
    selection = select_best_candidate(results)

    print(
        f"\nIncumbent ({INCUMBENT_CANDIDATE}) on the SAME {selection.n_evaluation_rows} "
        f"evaluation rows: {incumbent.mean_log_loss:.4f}"
    )
    print("Per-season held-out log loss:")
    for result in results:
        per_season = "  ".join(f"{s}={v:.4f}" for s, v in result.per_season_log_loss.items())
        print(f"  {result.name:22s} {per_season}")

    print("\nSelection-rule verdicts (all four conditions must hold to replace the incumbent):")
    for name, verdict in selection.reasons.items():
        comparison = verdict["paired_comparison"]
        print(f"  {name}:")
        print(f"    lower mean log loss            : {verdict['lower_mean_log_loss']}")
        print(f"    paired improvement, not a tie  : {verdict['paired_improvement_not_a_tie']} "
              f"(mean_diff={comparison['mean_diff']:+.5f}, SE={comparison['standard_error']:.5f}, "
              f"n={comparison['n']}, is_tie={comparison['is_tie']})")
        print(f"    improves every season          : {verdict['improves_every_evaluation_season']}")
        print(f"    no worst-season regression     : {verdict['no_worst_season_regression']}")
        print(f"    => PASSES ALL                  : {verdict['passes_all_conditions']}")

    print(f"\nSELECTED: {selection.selected}")
    if selection.incumbent_retained:
        print("  The raw strength trio is RETAINED - no challenger met every condition.")

    final_params = fit_final_meta_parameters(selection, predictions)
    print(f"\nFinal meta-parameters (fitted on all {EXPECTED_ROWS} development rows): {final_params}")
    if selection.incumbent_retained:
        print("  (identity - nothing to fit)")
    print(f"  These are RECORDED ONLY. They are not applied to {SEALED_SEASON} here.")

    print(
        f"\nHistorical context only: the frozen three-season trio figure is "
        f"{HISTORICAL_TRIO_THREE_SEASON_LOG_LOSS:.4f} over {EXPECTED_ROWS} rows, "
        f"which is NOT comparable to the {selection.n_evaluation_rows}-row "
        f"selection pool above."
    )

    _write_reports(predictions, results, selection, final_params)
    print(f"\nSealed season ({SEALED_SEASON}) scored: NO")
    return 0


def _write_reports(predictions, results, selection, final_params) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics_payload = [
        {"candidate": result.name, **fold_result.to_dict()}
        for result in results
        for fold_result in result.fold_results
    ]
    (REPORTS_DIR / "calibration_fold_metrics.json").write_text(
        json.dumps(fold_metrics_payload, indent=2, sort_keys=True) + "\n"
    )

    reliability = {}
    for result in results:
        reliability[result.name] = {
            fold_result.evaluation_season: per_class_reliability(
                fold_result.y_true, fold_result.proba
            )
            for fold_result in result.fold_results
        }

    summary_payload = {
        "stage": "calibration_ensemble_meta_layer",
        "evaluation_kind": (
            "DEVELOPMENT meta-validation (held out from each meta-parameter fit, but "
            "development labels were already inspected during design). The genuinely "
            "sealed final test remains the sealed season."
        ),
        "sealed_season": SEALED_SEASON,
        "sealed_season_scored": False,
        "frozen_base_models": {
            "outcome": "baseline_strength_trio",
            "scoreline": "dixon_coles_l2_decay",
        },
        "meta_protocol": "strict_chronological_expanding_window",
        "meta_folds": {
            str(f): META_FOLD_DEFINITIONS[f] for f in META_FOLDS
        },
        "calibration_seed_season": "2022_23",
        "n_development_rows": EXPECTED_ROWS,
        "n_evaluation_rows": EXPECTED_EVALUATION_ROWS,
        "candidates": {name: c.description for name, c in CANDIDATES.items()},
        "results": [result.to_dict() for result in results],
        "incumbent": INCUMBENT_CANDIDATE,
        "incumbent_mean_log_loss_on_evaluation_pool": selection.incumbent_mean_log_loss,
        "selection_verdicts": selection.reasons,
        "selected_candidate": selection.selected,
        "incumbent_retained": selection.incumbent_retained,
        "final_meta_parameters": final_params,
        "final_meta_parameters_applied_to_sealed_season": False,
        "historical_three_season_trio_log_loss": HISTORICAL_TRIO_THREE_SEASON_LOG_LOSS,
        "historical_figure_note": (
            "Computed over all 1,140 development rows across three seasons. Reporting "
            "context only - never a selection target, because selection happens on the "
            "760-row chronological evaluation pool."
        ),
        "selection_rule": (
            "A challenger replaces the raw trio only if ALL hold on the same 760 "
            "evaluation rows: (1) lower mean log loss; (2) paired per-match comparison "
            "is not a tie under the 2*SE rule and favours the challenger; (3) improves "
            "both 2023_24 and 2024_25; (4) no worst-season regression. Otherwise the "
            "raw trio is retained. Accuracy, Brier, macro-F1 and ECE are reporting "
            "metrics only."
        ),
        "per_class_reliability": reliability,
        "library_versions": library_versions(),
    }
    (REPORTS_DIR / "calibration_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, default=str) + "\n"
    )

    rows = []
    for result in results:
        for fold_result in result.fold_results:
            evaluate = predictions.subset([fold_result.evaluation_season])
            frame = evaluate.frame
            for i in range(len(frame)):
                rows.append(
                    {
                        "meta_fold": fold_result.meta_fold,
                        "meta_train_seasons": "+".join(fold_result.meta_train_seasons),
                        "Season": frame.iloc[i]["Season"],
                        "Date": frame.iloc[i]["Date"],
                        "HomeTeam": frame.iloc[i]["HomeTeam"],
                        "AwayTeam": frame.iloc[i]["AwayTeam"],
                        "actual_target": int(fold_result.y_true[i]),
                        "actual_ftr": frame.iloc[i]["actual_ftr"],
                        "p_home": float(fold_result.proba[i, 0]),
                        "p_draw": float(fold_result.proba[i, 1]),
                        "p_away": float(fold_result.proba[i, 2]),
                        "candidate": result.name,
                    }
                )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "calibration_predictions.csv", index=False)

    print(f"\nWrote reports to {REPORTS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
