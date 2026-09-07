"""Build and persist the FROZEN `baseline_strength_trio` inference artifact.

Model-selection and calibration/ensemble decisions are complete. This script
fits that frozen model ONCE, on every match through 2024/25 inclusive, and
serializes it. This IS the model that will eventually be applied exactly once
to the sealed 2025/26 season, once the rest of the project is frozen - it is
not a temporary artifact to be retrained or replaced before that evaluation
(see `StrengthTrioArtifact`'s docstring in `explanations.py` for the full
lifecycle).

Running this script does NOT open or score 2025/26: it reads only
`explanations.build_strength_trio_artifact`, which reads a feature file whose
row cap stops at 2024/25 and is structurally incapable of containing
sealed-season rows - the training cutoff is a module constant, not something
this script can override.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.explanations import (
    SEALED_SEASON,
    STRENGTH_TRIO_ARTIFACT_PATH,
    build_strength_trio_artifact,
    save_strength_trio_artifact,
)


def main(argv: list[str] | None = None) -> int:
    print("Building the FROZEN baseline_strength_trio inference artifact ...")
    artifact = build_strength_trio_artifact()

    meta = artifact.metadata
    print(f"  model_id: {meta.model_id}")
    print(f"  training_cutoff_season: {meta.training_cutoff_season}")
    print(f"  training_seasons: {meta.training_seasons}")
    print(f"  n_training_rows: {meta.n_training_rows}")
    print(f"  feature_columns: {meta.feature_columns}")
    print(f"  transformed_feature_names: {meta.transformed_feature_names}")
    print(f"  source_feature_artifact: {meta.source_feature_artifact}")
    print(f"  intended_final_evaluation_season: {meta.intended_final_evaluation_season}")
    print(f"  requires_retraining_before_final_evaluation: {meta.requires_retraining_before_final_evaluation}")
    print(f"  sealed season ({SEALED_SEASON}) in training_seasons: {SEALED_SEASON in meta.training_seasons}")

    save_strength_trio_artifact(artifact)
    print(f"\nWrote artifact to {STRENGTH_TRIO_ARTIFACT_PATH}")
    print(f"Wrote metadata sidecar to {STRENGTH_TRIO_ARTIFACT_PATH}.json")
    print(f"\nSealed season ({SEALED_SEASON}) scored: NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
