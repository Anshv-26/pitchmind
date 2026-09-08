"""Build and persist the FROZEN Dixon-Coles scoreline artifact.

Fits `dixon_coles_l2_decay` (the configuration frozen by the completed
model-selection stage) ONCE on 2015/16-2024/25 and writes it to
`models/dixon_coles_l2_decay_through_2024_25.json`.

    python scripts/build_score_model_artifact.py

Takes no training-data arguments: the source dataset, the frozen config id and
the 2024/25 cutoff are all module constants in
`backend.app.ml.score_model_artifact`, so this script cannot be pointed at a
wider window. Running it does NOT open or score the sealed 2025/26 season - the
sealed rows are filtered out and their absence is asserted before any fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.feature_engineering import SEALED_SEASON
from backend.app.ml.score_model_artifact import (
    SCORE_MODEL_ARTIFACT_PATH,
    build_score_model_artifact,
    save_score_model_artifact,
)


def main(argv: list[str] | None = None) -> int:
    print("Building the FROZEN dixon_coles_l2_decay scoreline artifact ...")
    artifact = build_score_model_artifact()

    meta = artifact.metadata
    print(f"  model_id: {meta.model_id} ({meta.model_type})")
    print(f"  source_dataset: {meta.source_dataset}")
    print(f"  training_cutoff_season: {meta.training_cutoff_season}")
    print(f"  training_seasons: {meta.training_seasons[0]}..{meta.training_seasons[-1]} "
          f"({len(meta.training_seasons)} seasons)")
    print(f"  training_match_count: {meta.training_match_count}")
    print(f"  use_dixon_coles: {meta.use_dixon_coles} | l2_sigma: {meta.l2_sigma} | "
          f"decay_half_life_days: {meta.decay_half_life_days}")
    print(f"  fitted teams: {len(artifact.params.teams)}")
    print(f"  fitted rho: {artifact.params.rho:.6f}")
    print(f"  home_advantage: {artifact.params.home_advantage:.6f} | "
          f"intercept: {artifact.params.intercept:.6f}")
    print(f"  sealed season ({SEALED_SEASON}) in training_seasons: "
          f"{SEALED_SEASON in meta.training_seasons}")

    save_score_model_artifact(artifact)
    print(f"\nWrote artifact to {SCORE_MODEL_ARTIFACT_PATH}")
    print(f"\nSealed season ({SEALED_SEASON}) scored: NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
