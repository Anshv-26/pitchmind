"""Mandatory, validated dataset loading for PitchMind model training.

`load_fold` is the ONLY sanctioned way for training code to turn a causal
feature artifact into model-ready X/y. It always calls
`feature_engineering.assert_artifact_valid_for` with its strict defaults
(never a bypass flag) before touching any row, so a training script cannot
accidentally fit or score on the canonical artifact, the wrong fold/evaluation
pairing, or a schedule with a broken causal-ordering invariant.

Only the three approved rolling-origin development folds are supported here.
There is no path to the sealed 2025/26 test season through this module - by
construction, not by convention: `load_fold` takes a fold NUMBER, never a raw
path, and its fold table stops at fold 3. The final fold (train through
2024/25, test 2025/26) is implemented separately, only after every modelling
decision is frozen (see the approved Stage 1 plan, section 17).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.app.ml.feature_engineering import (
    EloParamSchedule,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RESULT_COLUMN,
    SEALED_SEASON,
    TARGET_COLUMN,
    assert_artifact_valid_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# The three approved rolling-origin development folds. Fold 4 (train through
# 2024/25, test 2025/26) is deliberately absent.
DEVELOPMENT_FOLD_DEFINITIONS: dict[int, dict[str, str]] = {
    1: {"training_cutoff": "2021_22", "validation_season": "2022_23"},
    2: {"training_cutoff": "2022_23", "validation_season": "2023_24"},
    3: {"training_cutoff": "2023_24", "validation_season": "2024_25"},
}
DEVELOPMENT_FOLDS: tuple[int, ...] = tuple(sorted(DEVELOPMENT_FOLD_DEFINITIONS))

# Columns carried alongside X/y for debugging, joins, and the Stage 4
# out-of-fold prediction pool (Season/Date/HomeTeam/AwayTeam/actual result).
_METADATA_EXPORT_COLUMNS = METADATA_COLUMNS + [RESULT_COLUMN]


def artifact_path_for_fold(fold_number: int) -> Path:
    """The causal feature artifact path for a development fold, by construction.

    There is no way to pass an arbitrary path into `load_fold` - the artifact
    is always derived from the approved fold table, so training code can
    never accidentally point this at the canonical (`features.parquet`) or a
    diagnostic (`features_through_*.parquet`, fold-wide-estimation) artifact.
    """
    if fold_number not in DEVELOPMENT_FOLD_DEFINITIONS:
        raise ValueError(
            f"unknown development fold {fold_number!r}; only "
            f"{DEVELOPMENT_FOLDS} are supported here. The final fold (train "
            f"through 2024/25, test 2025/26) is intentionally not available "
            f"through this loader - the sealed season must never be touched "
            f"during development."
        )
    cutoff = DEVELOPMENT_FOLD_DEFINITIONS[fold_number]["training_cutoff"]
    return PROCESSED_DIR / f"features_causal_through_{cutoff}.parquet"


@dataclass(frozen=True)
class FoldData:
    """Everything a training/evaluation step needs for one development fold.

    `X_train`/`X_val` always carry the full `FEATURE_COLUMNS` (35 columns,
    never metadata, never the target). A model that wants the smaller
    `LINEAR_SAFE_FEATURE_COLUMNS` subset selects it itself
    (e.g. `fold.X_train[LINEAR_SAFE_FEATURE_COLUMNS]`) - this loader never
    bakes a model-specific column choice into the data it returns.
    """

    fold: int
    train_seasons: list[str]
    validation_season: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    metadata_train: pd.DataFrame
    metadata_val: pd.DataFrame
    artifact_path: Path
    feature_columns: list[str]
    elo_schedule: EloParamSchedule


def load_fold(fold_number: int) -> FoldData:
    """Load one development fold's train/validation split, safely.

    Before any row is read for modelling, this calls
    `assert_artifact_valid_for(artifact_path, [validation_season])` with its
    strict defaults - `require_intended_pairing=True` and
    `check_contents=True` are never relaxed here. That call alone rejects:
    the canonical (not-for-evaluation) artifact, a schedule with any
    non-causal entry, a schedule/evaluation-season pairing that is not the
    approved one, and an artifact whose row count or season contents disagree
    with its own provenance sidecar.

    On top of that shared guard, this function enforces the fold-specific
    invariants: no sealed-season rows anywhere in the artifact, exactly one
    validation season, and every training row strictly earlier (by date)
    than every validation row.
    """
    artifact_path = artifact_path_for_fold(fold_number)
    validation_season = DEVELOPMENT_FOLD_DEFINITIONS[fold_number]["validation_season"]

    # Mandatory safety gate. Never called with require_intended_pairing=False
    # or check_contents=False in training code.
    schedule = assert_artifact_valid_for(artifact_path, [validation_season])
    if not isinstance(schedule, EloParamSchedule):
        raise TypeError(
            f"{artifact_path} validated as a diagnostic single-EloParams "
            f"artifact, not a causal schedule. Development fold artifacts "
            f"must be built with --causal-through (see "
            f"scripts/build_features.py)."
        )

    frame = pd.read_parquet(artifact_path)

    present_seasons = set(frame["Season"].unique())
    if SEALED_SEASON in present_seasons:
        raise ValueError(
            f"fold {fold_number} artifact {artifact_path} contains "
            f"sealed-season ({SEALED_SEASON!r}) rows; refusing to use it for "
            f"development."
        )

    validation_mask = frame["Season"] == validation_season
    if int(validation_mask.sum()) == 0:
        raise ValueError(
            f"fold {fold_number} artifact {artifact_path} has no rows for "
            f"validation season {validation_season!r}."
        )
    validation_seasons_present = set(frame.loc[validation_mask, "Season"].unique())
    if validation_seasons_present != {validation_season}:
        raise ValueError(
            f"expected exactly one validation season {validation_season!r} "
            f"in {artifact_path}, found {validation_seasons_present}."
        )

    train_mask = ~validation_mask
    if validation_season in set(frame.loc[train_mask, "Season"].unique()):
        raise ValueError(
            f"validation season {validation_season!r} rows leaked into the "
            f"training split for fold {fold_number}."
        )

    missing_features = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing_features:
        raise ValueError(f"artifact {artifact_path} is missing features: {missing_features}")

    train_frame = frame.loc[train_mask].reset_index(drop=True)
    val_frame = frame.loc[validation_mask].reset_index(drop=True)

    if train_frame["Date"].max() >= val_frame["Date"].min():
        raise ValueError(
            f"fold {fold_number}: training rows are not strictly earlier "
            f"than validation rows (train max date "
            f"{train_frame['Date'].max()} >= validation min date "
            f"{val_frame['Date'].min()}). This should be impossible for a "
            f"correctly row-capped causal artifact - refusing to proceed."
        )

    return FoldData(
        fold=fold_number,
        train_seasons=sorted(train_frame["Season"].unique().tolist()),
        validation_season=validation_season,
        X_train=train_frame[FEATURE_COLUMNS].reset_index(drop=True),
        y_train=train_frame[TARGET_COLUMN].reset_index(drop=True),
        X_val=val_frame[FEATURE_COLUMNS].reset_index(drop=True),
        y_val=val_frame[TARGET_COLUMN].reset_index(drop=True),
        metadata_train=train_frame[_METADATA_EXPORT_COLUMNS].reset_index(drop=True),
        metadata_val=val_frame[_METADATA_EXPORT_COLUMNS].reset_index(drop=True),
        artifact_path=artifact_path,
        feature_columns=list(FEATURE_COLUMNS),
        elo_schedule=schedule,
    )


def load_all_development_folds() -> list[FoldData]:
    """Convenience: every development fold, in order."""
    return [load_fold(fold_number) for fold_number in DEVELOPMENT_FOLDS]
