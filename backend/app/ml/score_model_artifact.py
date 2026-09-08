"""Persistence for the FROZEN Dixon-Coles scoreline model.

The model-selection stage froze `dixon_coles_l2_decay` (Dixon-Coles, L2
sigma=0.25, 365-day half-life decay) as PitchMind's scoreline / expected-goals
model. This module fits that exact configuration ONCE, persists it, and loads
it read-only for serving. It contains no Dixon-Coles mathematics of its own -
every number comes from `ml.score_models`, which is frozen and untouched.

SERIALIZATION: JSON, not joblib
-------------------------------
A fitted `ScoreModelParams` is nothing but strings and floats (team names,
intercept, home advantage, per-team attack/defence, rho, promoted offsets), so
it round-trips through plain JSON. That is strictly better than a pickle-based
format here: the artifact is human-readable, diffable, and carries no
arbitrary-code-execution risk on load. (The strength-trio artifact needs joblib
only because it holds live scikit-learn transformer objects.)

SEALED-SEASON BOUNDARY
----------------------
The training source is `data/processed/matches.parquet` filtered to exclude
`SEALED_SEASON`. That filter is NOT a convenience - it is load-bearing, and it
is asserted rather than trusted.

The obvious-looking alternative does not work, and the reason is worth
recording: `features_causal_through_2023_24.parquet` is structurally
sealed-season-free (its row cap stops at 2024/25), but it carries engineered
features WITHOUT `FTHG`/`FTAG`, and a goals model cannot be fitted without
goals. `matches.parquet` has the goals but includes 2025/26. So the only viable
source is the one that must be filtered explicitly - hence the assertions
below, which fail loudly rather than quietly training on a sealed match.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from backend.app.ml.feature_engineering import SEALED_SEASON
from backend.app.ml.score_models import (
    CONFIGS,
    MATCH_COLUMNS,
    ScoreModelConfig,
    ScoreModelParams,
    fit_score_model,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MATCHES_PATH = REPO_ROOT / "data" / "processed" / "matches.parquet"

# The frozen scoreline configuration. Not a choice made here - the
# model-selection stage decided it, and this module only serves it.
FROZEN_SCORE_MODEL_CONFIG_ID = "dixon_coles_l2_decay"

# The model is fitted on every match through this season inclusive.
TRAINING_CUTOFF_SEASON = "2024_25"

SCORE_MODEL_ARTIFACT_FORMAT_VERSION = "1.0"
SCORE_MODEL_ARTIFACT_PATH = (
    REPO_ROOT / "models" / "dixon_coles_l2_decay_through_2024_25.json"
)
BUILD_SCRIPT = "scripts/build_score_model_artifact.py"


def library_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


@dataclass(frozen=True)
class ScoreModelArtifactMetadata:
    """Provenance recorded alongside the fitted parameters."""

    model_id: str
    model_type: str
    artifact_format_version: str
    training_cutoff_season: str
    training_seasons: list[str]
    training_match_count: int
    use_dixon_coles: bool
    l2_sigma: float | None
    decay_half_life_days: float | None
    source_dataset: str
    created_by_script: str
    intended_final_evaluation_season: str
    requires_retraining_before_final_evaluation: bool
    sealed_final_test_scored: bool
    library_versions: dict[str, str]


@dataclass(frozen=True)
class ScoreModelArtifact:
    """The frozen Dixon-Coles model, ready for inference.

    Like the strength-trio artifact, this IS the model intended for eventual
    one-time application to the sealed season - not a temporary object to be
    refitted first.
    """

    metadata: ScoreModelArtifactMetadata
    params: ScoreModelParams
    config: ScoreModelConfig


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_score_model_artifact() -> ScoreModelArtifact:
    """Fit the frozen Dixon-Coles configuration on 2015/16-2024/25 ONLY.

    Takes no arguments: the config id, training cutoff and source dataset are
    all module constants, so there is no parameter through which a caller
    could widen the training window or slip the sealed season in.
    """
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(f"{MATCHES_PATH} not found; run scripts/process_data.py first")

    frame = pd.read_parquet(MATCHES_PATH, columns=MATCH_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"])

    # Load-bearing filter, then assert it actually worked.
    train = frame.loc[frame["Season"] != SEALED_SEASON].reset_index(drop=True)
    if SEALED_SEASON in set(train["Season"].unique()):
        raise RuntimeError(
            f"sealed season {SEALED_SEASON!r} survived filtering; refusing to fit on it"
        )
    if len(train) == 0:
        raise RuntimeError("no training rows remain after excluding the sealed season")

    training_seasons = sorted(train["Season"].unique().tolist())
    if training_seasons[-1] != TRAINING_CUTOFF_SEASON:
        raise RuntimeError(
            f"expected training to end at {TRAINING_CUTOFF_SEASON!r}, got "
            f"{training_seasons[-1]!r} (seasons: {training_seasons})"
        )

    config = CONFIGS[FROZEN_SCORE_MODEL_CONFIG_ID]
    params = fit_score_model(train, config)

    metadata = ScoreModelArtifactMetadata(
        model_id=FROZEN_SCORE_MODEL_CONFIG_ID,
        model_type="dixon_coles",
        artifact_format_version=SCORE_MODEL_ARTIFACT_FORMAT_VERSION,
        training_cutoff_season=TRAINING_CUTOFF_SEASON,
        training_seasons=training_seasons,
        training_match_count=len(train),
        use_dixon_coles=config.use_dixon_coles,
        l2_sigma=config.l2_sigma,
        decay_half_life_days=config.half_life_days,
        source_dataset=MATCHES_PATH.name,
        created_by_script=BUILD_SCRIPT,
        intended_final_evaluation_season=SEALED_SEASON,
        requires_retraining_before_final_evaluation=False,
        sealed_final_test_scored=False,
        library_versions=library_versions(),
    )
    return ScoreModelArtifact(metadata=metadata, params=params, config=config)


# --------------------------------------------------------------------------
# Serialize / deserialize (plain JSON - no pickle)
# --------------------------------------------------------------------------
def _params_to_dict(params: ScoreModelParams) -> dict[str, Any]:
    return {
        "config_id": params.config_id,
        "teams": list(params.teams),
        "intercept": params.intercept,
        "home_advantage": params.home_advantage,
        "attack": dict(params.attack),
        "defence": dict(params.defence),
        "rho": params.rho,
        "promoted_attack_offset": params.promoted_attack_offset,
        "promoted_defence_offset": params.promoted_defence_offset,
    }


def _params_from_dict(raw: dict[str, Any]) -> ScoreModelParams:
    return ScoreModelParams(
        config_id=raw["config_id"],
        teams=tuple(raw["teams"]),
        intercept=float(raw["intercept"]),
        home_advantage=float(raw["home_advantage"]),
        attack={k: float(v) for k, v in raw["attack"].items()},
        defence={k: float(v) for k, v in raw["defence"].items()},
        rho=None if raw["rho"] is None else float(raw["rho"]),
        promoted_attack_offset=float(raw["promoted_attack_offset"]),
        promoted_defence_offset=float(raw["promoted_defence_offset"]),
    )


def _config_to_dict(config: ScoreModelConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "use_dixon_coles": config.use_dixon_coles,
        "l2_sigma": config.l2_sigma,
        "half_life_days": config.half_life_days,
    }


def _config_from_dict(raw: dict[str, Any]) -> ScoreModelConfig:
    return ScoreModelConfig(
        config_id=raw["config_id"],
        use_dixon_coles=bool(raw["use_dixon_coles"]),
        l2_sigma=None if raw["l2_sigma"] is None else float(raw["l2_sigma"]),
        half_life_days=None if raw["half_life_days"] is None else float(raw["half_life_days"]),
    )


def _validate_artifact_contract(artifact: ScoreModelArtifact) -> None:
    """Fail loudly on any contract mismatch. Called on every load."""
    meta = artifact.metadata
    if meta.model_id != FROZEN_SCORE_MODEL_CONFIG_ID:
        raise ValueError(f"unexpected model_id {meta.model_id!r} in score-model artifact")
    if meta.artifact_format_version != SCORE_MODEL_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"score-model artifact format version {meta.artifact_format_version!r} does not "
            f"match the version this code expects ({SCORE_MODEL_ARTIFACT_FORMAT_VERSION!r})"
        )
    if meta.training_cutoff_season != TRAINING_CUTOFF_SEASON:
        raise ValueError(
            f"score-model artifact training cutoff {meta.training_cutoff_season!r} does not "
            f"match the expected {TRAINING_CUTOFF_SEASON!r}"
        )
    if SEALED_SEASON in meta.training_seasons:
        raise ValueError(
            f"score-model artifact lists the sealed season {SEALED_SEASON!r} as a training season"
        )
    if meta.sealed_final_test_scored:
        raise ValueError("score-model artifact claims the sealed final test has been scored")
    if artifact.params.config_id != FROZEN_SCORE_MODEL_CONFIG_ID:
        raise ValueError(f"fitted params are for {artifact.params.config_id!r}, not the frozen config")
    if not artifact.config.use_dixon_coles:
        raise ValueError("frozen scoreline model must be a Dixon-Coles configuration")
    if not artifact.params.teams:
        raise ValueError("score-model artifact contains no fitted teams")


def save_score_model_artifact(
    artifact: ScoreModelArtifact, path: Path = SCORE_MODEL_ARTIFACT_PATH
) -> None:
    """Persist as readable JSON."""
    _validate_artifact_contract(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": asdict(artifact.metadata),
        "config": _config_to_dict(artifact.config),
        "params": _params_to_dict(artifact.params),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_score_model_artifact(
    path: Path = SCORE_MODEL_ARTIFACT_PATH,
) -> ScoreModelArtifact:
    """Load a previously-built artifact. Never fits or refits anything."""
    if not path.exists():
        raise FileNotFoundError(
            f"score-model artifact not found; run {BUILD_SCRIPT} to create it"
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"score-model artifact is not valid JSON: {exc}") from exc

    try:
        artifact = ScoreModelArtifact(
            metadata=ScoreModelArtifactMetadata(**payload["metadata"]),
            params=_params_from_dict(payload["params"]),
            config=_config_from_dict(payload["config"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"score-model artifact is missing required fields: {exc}") from exc

    _validate_artifact_contract(artifact)
    return artifact


__all__ = [
    "FROZEN_SCORE_MODEL_CONFIG_ID",
    "SCORE_MODEL_ARTIFACT_PATH",
    "SCORE_MODEL_ARTIFACT_FORMAT_VERSION",
    "TRAINING_CUTOFF_SEASON",
    "ScoreModelArtifact",
    "ScoreModelArtifactMetadata",
    "build_score_model_artifact",
    "load_score_model_artifact",
    "save_score_model_artifact",
]
