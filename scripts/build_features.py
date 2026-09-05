"""Build the pre-match feature dataset from the clean match dataset.

Thin CLI only: all feature logic lives in
``backend/app/ml/feature_engineering.py``. Reads
``data/processed/matches.parquet``, writes a feature parquet plus a provenance
sidecar, and prints a concise report. Nothing under ``data/raw/`` or
``matches.parquet`` is ever written to.

Three build modes, each writing to a distinct path so one can never silently
overwrite another:

* no flags → ``data/processed/features.parquet``, the **canonical**
  artifact, built with a-priori placeholder Elo parameters and explicitly
  marked NOT VALID FOR MODEL EVALUATION.
* ``--causal-through 2021_22`` → ``data/processed/features_causal_through_2021_22.parquet``,
  the **normal mode for fold/final artifacts**. Builds the single canonical
  causal Elo schedule (every season's parameters estimated only from strictly
  earlier seasons — see ``build_causal_elo_schedule``), then writes only rows
  through the training cutoff's paired evaluation season (e.g. 2021_22 ->
  rows through 2022_23), so development artifacts structurally contain no
  sealed-season (2025/26) rows. ``--causal-through`` must be one of
  2021_22, 2022_23, 2023_24, 2024_25 (the last one produces the final
  artifact, which legitimately includes 2025/26).
* ``--estimate-through 2021_22`` → ``data/processed/features_through_2021_22.parquet``,
  the older **diagnostic** mode: a single ``EloParams`` object (fold-wide
  estimation) applied across the whole matches frame. Kept for
  backward-compatible/diagnostic use only — it is NOT causal at the parameter
  level (see the module docstring in ``feature_engineering.py``) and should
  not be used to build an artifact for model evaluation.

``--output`` overrides the path. An existing artifact whose provenance differs
from the new build is never overwritten without ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.feature_engineering import (  # noqa: E402
    CANONICAL_NOT_VALID_NOTE,
    CANONICAL_PURPOSE,
    DEFAULT_EFFICIENCY_WINDOW,
    DEFAULT_EWMA_HALFLIFE,
    DEFAULT_MIN_PERIODS,
    DEFAULT_REST_DAYS_CAP,
    FEATURE_COLUMNS,
    INTENDED_TRAINING_CUTOFF,
    LINEAR_SAFE_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MIN_PRIOR_SEASONS_FOR_ESTIMATION,
    SEALED_SEASON,
    TARGET_COLUMN,
    EloParamSchedule,
    active_season_mean_elo,
    build_causal_elo_schedule,
    build_features,
    build_provenance,
    canonical_elo_params,
    cold_start_row_count,
    elo_params_to_dict,
    estimate_elo_params,
    intended_evaluation_seasons_for,
    sidecar_path_for,
    validate_feature_frame,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SOURCE_PATH = PROCESSED_DIR / "matches.parquet"
CANONICAL_OUTPUT_PATH = PROCESSED_DIR / "features.parquet"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_output_path(
    estimate_through: str | None = None, *, causal_through: str | None = None
) -> Path:
    """Canonical, causal-fold, and diagnostic-fold artifacts get distinct paths."""
    if causal_through is not None:
        return PROCESSED_DIR / f"features_causal_through_{causal_through}.parquet"
    if estimate_through is None:
        return CANONICAL_OUTPUT_PATH
    return PROCESSED_DIR / f"features_through_{estimate_through}.parquet"


def season_order(matches: pd.DataFrame) -> list[str]:
    return list(matches.groupby("Season")["Date"].min().sort_values().index)


def refuse_incompatible_overwrite(output_path: Path, new_provenance: dict, force: bool) -> str | None:
    """Return an error message if overwriting would destroy different provenance."""
    sidecar = sidecar_path_for(output_path)
    if force or not output_path.exists() or not sidecar.exists():
        return None
    try:
        existing = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return (
            f"{output_path} exists but its sidecar {sidecar} is unreadable. "
            f"Refusing to overwrite; pass --force to replace it."
        )
    same_provenance = (
        existing.get("estimated_from_seasons") == new_provenance["estimated_from_seasons"]
        and existing.get("valid_for_model_evaluation")
        == new_provenance["valid_for_model_evaluation"]
        and existing.get("artifact_row_cap_season") == new_provenance.get("artifact_row_cap_season")
    )
    if same_provenance:
        return None
    return (
        f"{output_path} already holds an artifact with different provenance "
        f"(existing training cutoff: {existing.get('training_cutoff')!r}, "
        f"new: {new_provenance.get('training_cutoff')!r}). Refusing to replace "
        f"it silently — choose a different --output or pass --force."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the pre-match feature dataset from matches.parquet"
    )
    parser.add_argument(
        "--causal-through",
        metavar="SEASON",
        choices=sorted(set(INTENDED_TRAINING_CUTOFF.values())),
        help=(
            "Build the CAUSAL production artifact for the fold/final build "
            "whose training cutoff is SEASON. Every season's rows are "
            "generated with Elo parameters estimated only from strictly "
            "earlier seasons; output rows are capped at this cutoff's paired "
            "evaluation season, so development artifacts structurally "
            "contain no sealed-season rows. This is the normal mode for "
            "fold and final evaluation artifacts."
        ),
    )
    parser.add_argument(
        "--estimate-through",
        metavar="SEASON",
        help=(
            "DIAGNOSTIC mode: estimate ONE fixed EloParams object from seasons "
            "up to and including SEASON and apply it across the whole matches "
            "frame. Not causal at the parameter level - do not use this to "
            "build an artifact for model evaluation. Prefer --causal-through."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output parquet path (default depends on build mode)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing artifact that has different provenance",
    )
    args = parser.parse_args(argv)

    if args.causal_through and args.estimate_through:
        print("FAILED: pass only one of --causal-through / --estimate-through")
        return 1

    if not SOURCE_PATH.exists():
        print(f"FAILED: {SOURCE_PATH} not found (run scripts/process_data.py first)")
        return 1

    matches = pd.read_parquet(SOURCE_PATH)
    print(f"Loaded {len(matches)} matches from {SOURCE_PATH}")

    row_cap_season: str | None = None

    if args.causal_through:
        row_cap_seasons = intended_evaluation_seasons_for(args.causal_through)
        if not row_cap_seasons:
            print(
                f"FAILED: {args.causal_through!r} has no approved evaluation "
                f"season in INTENDED_TRAINING_CUTOFF"
            )
            return 1
        row_cap_season = row_cap_seasons[0]

        try:
            schedule = build_causal_elo_schedule(matches)
        except ValueError as exc:
            print(f"FAILED: could not build the causal Elo schedule: {exc}")
            return 1

        print(f"Built causal Elo schedule ({len(schedule.seasons)} seasons, "
              f"causal={schedule.is_causal()})")

        full_features = build_features(matches, schedule)
        order = season_order(matches)
        seasons_to_keep = set(order[: order.index(row_cap_season) + 1])
        features = full_features[full_features["Season"].isin(seasons_to_keep)].reset_index(
            drop=True
        )
        expected_rows = int(matches["Season"].isin(seasons_to_keep).sum())

        elo_config = schedule
        intended = [row_cap_season]
        purpose = (
            f"Causal fold/evaluation artifact: training cutoff "
            f"{args.causal_through}, row-capped at {row_cap_season}."
        )
        valid_for_evaluation = True
        earliest_valid = row_cap_season
        notes = (
            f"Intended for evaluating {row_cap_season} only. Call "
            f"assert_artifact_valid_for() before any fit or scoring."
        )

    elif args.estimate_through:
        try:
            elo_config = estimate_elo_params(matches, args.estimate_through)
        except ValueError as exc:
            print(f"FAILED: could not estimate Elo parameters: {exc}")
            return 1
        intended = intended_evaluation_seasons_for(args.estimate_through)
        purpose = (
            f"DIAGNOSTIC fold-wide artifact (not causal at the parameter "
            f"level): Elo parameters estimated from seasons through "
            f"{args.estimate_through}."
        )
        valid_for_evaluation = True
        earliest_valid = _next_season(matches, args.estimate_through)
        notes = (
            f"Intended for evaluating {intended or '(no approved fold)'} only. "
            f"Fold-wide estimation, not causal at the parameter level - prefer "
            f"--causal-through for real evaluation. Call "
            f"assert_artifact_valid_for() before any fit or scoring."
        )
        features = build_features(matches, elo_config)
        expected_rows = len(matches)

    else:
        elo_config = canonical_elo_params()
        intended = []
        purpose = CANONICAL_PURPOSE
        valid_for_evaluation = False
        earliest_valid = None
        notes = CANONICAL_NOT_VALID_NOTE
        features = build_features(matches, elo_config)
        expected_rows = len(matches)

    problems = validate_feature_frame(
        features, expected_rows=expected_rows, initial_rating=elo_config.initial_rating
    )
    if problems:
        print()
        print(f"VALIDATION FAILED - {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print("No output file written.")
        return 1

    provenance = build_provenance(
        elo_params=elo_config,
        ewma_halflife=DEFAULT_EWMA_HALFLIFE,
        min_periods=DEFAULT_MIN_PERIODS,
        efficiency_window=DEFAULT_EFFICIENCY_WINDOW,
        rest_days_cap=DEFAULT_REST_DAYS_CAP,
        source_path=str(SOURCE_PATH.relative_to(REPO_ROOT)),
        source_sha256=sha256_of(SOURCE_PATH),
        n_rows=len(features),
        purpose=purpose,
        valid_for_evaluation=valid_for_evaluation,
        earliest_valid_evaluation_season=earliest_valid,
        intended_evaluation_seasons=intended,
        artifact_row_cap_season=row_cap_season,
        min_prior_seasons_for_estimation=(
            MIN_PRIOR_SEASONS_FOR_ESTIMATION if args.causal_through else None
        ),
        notes=notes,
    )

    output_path = args.output if args.output is not None else default_output_path(
        args.estimate_through, causal_through=args.causal_through
    )
    refusal = refuse_incompatible_overwrite(output_path, provenance, args.force)
    if refusal:
        print()
        print(f"REFUSED: {refusal}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, engine="pyarrow", index=False)
    sidecar = sidecar_path_for(output_path)
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    _print_report(
        features, elo_config, output_path, sidecar, valid_for_evaluation, intended, row_cap_season
    )
    return 0


def _next_season(matches: pd.DataFrame, season: str) -> str | None:
    order = season_order(matches)
    if season not in order:
        return None
    index = order.index(season) + 1
    return order[index] if index < len(order) else None


def _print_report(
    features: pd.DataFrame,
    elo_config,
    output_path: Path,
    sidecar: Path,
    valid_for_evaluation: bool,
    intended: list[str],
    row_cap_season: str | None,
) -> None:
    print()
    print(f"Rows:     {len(features)}")
    print(
        f"Features: {len(FEATURE_COLUMNS)} "
        f"(linear-safe subset: {len(LINEAR_SAFE_FEATURE_COLUMNS)})   "
        f"Metadata: {len(METADATA_COLUMNS)}   Target: {TARGET_COLUMN}"
    )
    print(f"Seasons:  {features['Season'].nunique()}  "
          f"({features['Season'].min()} -> {features['Season'].max()})")
    print(f"Dates:    {features['Date'].min().date()} -> {features['Date'].max().date()}")
    if row_cap_season is not None:
        print(f"Row cap:  {row_cap_season} (no rows beyond this season)")

    print()
    if isinstance(elo_config, EloParamSchedule):
        print(f"Elo schedule ({len(elo_config.seasons)} seasons, causal={elo_config.is_causal()}):")
        for season in elo_config.seasons:
            p = elo_config.params_for(season)
            source = f"estimated<={p.training_cutoff}" if p.is_estimated else "a-priori"
            print(
                f"  {season}: {source:<16} ha={p.home_advantage:6.2f}  "
                f"shrink={p.season_shrink:.3f}  delta={p.promoted_prior_delta:7.2f}"
            )
    else:
        print("Elo parameters used (diagnostic, fold-wide, NOT causal per-row):")
        for key, value in elo_params_to_dict(elo_config).items():
            print(f"  {key}: {value}")

    print()
    if valid_for_evaluation:
        print(f"Artifact status: VALID for evaluating {intended or '(no approved fold)'}")
    else:
        print("Artifact status: *** NOT VALID FOR MODEL EVALUATION ***")
        print("  Built with a-priori placeholder Elo parameters (no provenance).")
        print("  Use --causal-through <fold train cutoff> to build a fold artifact.")

    print()
    cold = cold_start_row_count(features)
    print(f"Cold-start rows retained (either side < min history): {cold} ({cold / len(features):.2%})")
    sealed_rows = int((features["Season"] == SEALED_SEASON).sum())
    print(f"Sealed season present in artifact: {SEALED_SEASON} ({sealed_rows} rows)")

    print()
    print("Feature NaN counts (non-zero only):")
    nan_counts = features[FEATURE_COLUMNS].isna().sum()
    nonzero = nan_counts[nan_counts > 0]
    if nonzero.empty:
        print("  none")
    else:
        for column, count in nonzero.items():
            print(f"  {column}: {count}")

    print()
    print("Mean Elo of each club's season opener (scale stability check):")
    for season, mean_elo in active_season_mean_elo(features).items():
        print(f"  {season}: {mean_elo:.1f}")

    print()
    print(f"Elo range: {features['home_elo'].min():.1f} -> {features['home_elo'].max():.1f}")

    print()
    print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"Wrote {sidecar} ({sidecar.stat().st_size:,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
