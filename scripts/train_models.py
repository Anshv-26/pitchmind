"""Stage 1 development-only training CLI.

Runs baselines and (optionally) the four model families across the three
approved rolling-origin development folds, prints a concise leaderboard, and
writes lightweight JSON/CSV reports under reports/modeling/.

This script has NO path to the sealed 2025/26 season: `datasets.load_fold`
only supports the three development folds, and this script never accepts a
season or fold argument that could reach anything else. The assertion in
`main()` below is a redundant, explicit second check on top of that
structural guarantee.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.datasets import DEVELOPMENT_FOLDS, load_all_development_folds
from backend.app.ml.feature_engineering import SEALED_SEASON
from backend.app.ml.training import (
    MODEL_SPECS,
    library_versions,
    run_baseline_experiments,
    run_grid_experiments,
    select_best_configuration,
)

REPORTS_DIR = REPO_ROOT / "reports" / "modeling"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1 development training (development folds 1-3 only; never scores 2025/26)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Model families to run in addition to the baselines (default: all)",
    )
    parser.add_argument(
        "--baselines-only", action="store_true", help="Run only the three baselines, no model grids"
    )
    args = parser.parse_args(argv)

    print(f"Loading development folds {DEVELOPMENT_FOLDS} ...")
    folds = load_all_development_folds()

    # Redundant, explicit safety check (load_all_development_folds already
    # cannot reach 2025/26 - see datasets.DEVELOPMENT_FOLD_DEFINITIONS).
    for fold in folds:
        if fold.validation_season == SEALED_SEASON or SEALED_SEASON in fold.train_seasons:
            print(f"REFUSED: fold {fold.fold} touches the sealed season {SEALED_SEASON!r}.")
            return 1

    for fold in folds:
        print(
            f"  fold {fold.fold}: train {fold.train_seasons[0]}..{fold.train_seasons[-1]} "
            f"({len(fold.X_train)} rows) -> validate {fold.validation_season} ({len(fold.X_val)} rows)"
        )

    all_results = []
    all_predictions = []

    print("\nRunning baselines ...")
    baseline_results, baseline_predictions = run_baseline_experiments(folds)
    all_results.extend(baseline_results)
    all_predictions.extend(baseline_predictions)
    for result in baseline_results:
        print(f"  {result.model:28s} mean_logloss={result.mean_log_loss:.4f}  worst={result.worst_log_loss:.4f}")

    if not args.baselines_only:
        for model_name in args.models:
            spec = MODEL_SPECS[model_name]
            print(f"\nRunning {model_name} ({len(spec.param_grid)} configuration(s)) ...")
            grid_results, grid_predictions = run_grid_experiments(spec, folds)
            all_results.extend(grid_results)
            all_predictions.extend(grid_predictions)
            best_for_model = min(grid_results, key=lambda r: r.mean_log_loss)
            print(
                f"  best {model_name}: {best_for_model.config_id} "
                f"mean_logloss={best_for_model.mean_log_loss:.4f}"
            )

    best = select_best_configuration(all_results)
    print(f"\nSelected configuration: {best.config_id} (mean_logloss={best.mean_log_loss:.4f})")
    print("Per-season log loss for the selected configuration:")
    for metrics in best.fold_metrics:
        print(f"  fold {metrics.fold} ({metrics.validation_season}): {metrics.log_loss:.4f}")

    _write_reports(all_results, all_predictions, best)
    print(f"\nSealed season ({SEALED_SEASON}) scored: NO")
    return 0


def _write_reports(all_results, all_predictions, best) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics_payload = [
        {"model": result.model, "config_id": result.config_id, **metrics.to_dict()}
        for result in all_results
        for metrics in result.fold_metrics
    ]
    (REPORTS_DIR / "stage1_fold_metrics.json").write_text(
        json.dumps(fold_metrics_payload, indent=2, sort_keys=True) + "\n"
    )

    summary_payload = {
        "development_folds": list(DEVELOPMENT_FOLDS),
        "sealed_season": SEALED_SEASON,
        "sealed_season_scored": False,
        "results": [
            {k: v for k, v in result.to_dict().items() if k != "fold_metrics"} for result in all_results
        ],
        "selected_configuration": best.config_id,
        "selection_rule": (
            "lowest mean development log loss; ties (|paired mean diff| < 2*SE "
            "against the current best candidate, computed over pooled "
            "per-row development log-loss differences) broken by worst-season "
            "log loss, then model simplicity "
            "(logreg < random_forest < xgboost ~ catboost), then lower "
            "across-fold log-loss variance. Accuracy never used for selection."
        ),
        "library_versions": library_versions(),
    }
    (REPORTS_DIR / "stage1_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )

    predictions_frame = pd.DataFrame(all_predictions)
    predictions_frame.to_csv(REPORTS_DIR / "stage1_predictions.csv", index=False)

    print(f"\nWrote reports to {REPORTS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
