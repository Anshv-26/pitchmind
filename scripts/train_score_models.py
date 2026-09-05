"""Development-only training CLI for the Poisson / Dixon-Coles score models.

Runs the five pre-committed configurations across the three approved
rolling-origin development folds, prints a leaderboard (with per-season log
loss) alongside the frozen Stage 1 champion for direct comparison, and writes
lightweight JSON/CSV reports under reports/modeling/.

This script has NO path to the sealed 2025/26 season:
`score_models.load_all_development_score_folds` only ever returns the three
approved development folds (same fold table as Stage 1), and nothing here
accepts a season or fold argument that could reach anything else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.datasets import DEVELOPMENT_FOLDS
from backend.app.ml.feature_engineering import SEALED_SEASON
from backend.app.ml.score_models import (
    CONFIGS,
    library_versions,
    load_all_development_score_folds,
    run_score_model_experiment,
    select_best_score_model,
)

REPORTS_DIR = REPO_ROOT / "reports" / "modeling"
STAGE1_SUMMARY_PATH = REPORTS_DIR / "stage1_summary.json"


def _stage1_champion() -> tuple[str, float] | None:
    """Read the frozen Stage 1 champion for the leaderboard comparison line.
    Read-only; Stage 1 results are never modified or retrained here."""
    if not STAGE1_SUMMARY_PATH.exists():
        return None
    payload = json.loads(STAGE1_SUMMARY_PATH.read_text())
    selected = payload.get("selected_configuration")
    for result in payload.get("results", []):
        if result.get("config_id") == selected:
            return selected, result["mean_log_loss"]
    return None


def main(argv: list[str] | None = None) -> int:
    print(f"Loading development folds {DEVELOPMENT_FOLDS} ...")
    folds = load_all_development_score_folds()

    # Redundant, explicit safety check (the loader already cannot reach
    # 2025/26 - see score_models.load_score_fold).
    for fold in folds:
        seasons_touched = set(fold.train_seasons) | {fold.validation_season}
        if SEALED_SEASON in seasons_touched:
            print(f"REFUSED: fold {fold.fold} touches the sealed season {SEALED_SEASON!r}.")
            return 1

    for fold in folds:
        print(
            f"  fold {fold.fold}: train {fold.train_seasons[0]}..{fold.train_seasons[-1]} "
            f"({len(fold.train)} rows) -> validate {fold.validation_season} ({len(fold.validation)} rows)"
        )

    all_results = []
    all_predictions = []

    print(f"\nRunning {len(CONFIGS)} score-model configuration(s) ...")
    for config in CONFIGS.values():
        result, rows = run_score_model_experiment(config, folds)
        all_results.append(result)
        all_predictions.extend(rows)
        print(f"  {result.config_id:24s} mean_logloss={result.mean_log_loss:.4f}  worst={result.worst_log_loss:.4f}")

    best = select_best_score_model(all_results)
    print(f"\nSelected score-model configuration: {best.config_id} (mean_logloss={best.mean_log_loss:.4f})")
    print("Per-season log loss for the selected configuration:")
    for metrics in best.fold_metrics:
        print(f"  fold {metrics.fold} ({metrics.validation_season}): {metrics.log_loss:.4f}")

    champion = _stage1_champion()
    print()
    if champion is not None:
        champion_id, champion_ll = champion
        delta = best.mean_log_loss - champion_ll
        verdict = "BEATS" if delta < 0 else ("TIES" if abs(delta) < 1e-6 else "LOSES TO")
        print(f"Stage 1 champion: {champion_id} = {champion_ll:.4f}")
        print(f"Best score model: {best.config_id} = {best.mean_log_loss:.4f}  ({verdict} the champion by {delta:+.4f})")
    else:
        print("Stage 1 summary not found - skipping champion comparison.")

    _write_reports(all_results, all_predictions, best)
    print(f"\nSealed season ({SEALED_SEASON}) scored: NO")
    return 0


def _write_reports(all_results, all_predictions, best) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics_payload = [
        {"config_id": result.config_id, **metrics.to_dict()}
        for result in all_results
        for metrics in result.fold_metrics
    ]
    (REPORTS_DIR / "score_models_fold_metrics.json").write_text(
        json.dumps(fold_metrics_payload, indent=2, sort_keys=True) + "\n"
    )

    champion = _stage1_champion()
    summary_payload = {
        "development_folds": list(DEVELOPMENT_FOLDS),
        "sealed_season": SEALED_SEASON,
        "sealed_season_scored": False,
        "results": [
            {k: v for k, v in result.to_dict().items() if k != "fold_metrics"} for result in all_results
        ],
        "selected_configuration": best.config_id,
        "stage1_champion": (
            {"config_id": champion[0], "mean_log_loss": champion[1]} if champion is not None else None
        ),
        "selection_rule": (
            "lowest mean development log loss; ties (|paired mean diff| < 2*SE "
            "against the current best candidate, computed over pooled per-row "
            "development log-loss differences) broken by worst-season log loss, "
            "then config simplicity (poisson < poisson_l2 < dixon_coles < "
            "dixon_coles_l2 < dixon_coles_l2_decay), then lower across-fold "
            "log-loss variance. Accuracy never used for selection. This "
            "selection logic is LOCAL to score_models.py, not Stage 1's "
            "training.select_best_configuration."
        ),
        "library_versions": library_versions(),
    }
    (REPORTS_DIR / "score_models_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )

    predictions_frame = pd.DataFrame(all_predictions)
    predictions_frame.to_csv(REPORTS_DIR / "score_models_predictions.csv", index=False)

    print(f"\nWrote reports to {REPORTS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
