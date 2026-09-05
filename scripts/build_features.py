"""Build the pre-match feature dataset from the clean match dataset.

Thin CLI only: all feature logic lives in
``backend/app/ml/feature_engineering.py``. Reads
``data/processed/matches.parquet``, writes a feature parquet plus a provenance
sidecar, and prints a concise report. Nothing under ``data/raw/`` or
``matches.parquet`` is ever written to.

Two build modes, which write to *different* paths so a fold artifact can never
silently replace the canonical one:

* no ``--estimate-through`` → ``data/processed/features.parquet``, the
  **canonical** artifact, built with a-priori placeholder Elo parameters and
  explicitly marked NOT VALID FOR MODEL EVALUATION.
* ``--estimate-through 2021_22`` → ``data/processed/features_through_2021_22.parquet``,
  a **fold** artifact whose Elo parameters are estimated from that fold's
  training window only.

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
    LINEAR_SAFE_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    SEALED_SEASON,
    TARGET_COLUMN,
    active_season_mean_elo,
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


def default_output_path(estimate_through: str | None) -> Path:
    """Canonical and fold artifacts get distinct, non-colliding paths."""
    if estimate_through is None:
        return CANONICAL_OUTPUT_PATH
    return PROCESSED_DIR / f"features_through_{estimate_through}.parquet"


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
    same_provenance = existing.get("estimated_from_seasons") == new_provenance[
        "estimated_from_seasons"
    ] and existing.get("valid_for_model_evaluation") == new_provenance[
        "valid_for_model_evaluation"
    ]
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
        "--estimate-through",
        metavar="SEASON",
        help=(
            "Estimate Elo parameters from seasons up to and including SEASON "
            "(a fold's training cutoff), and write a distinct fold artifact. "
            "Omit to build the canonical artifact with a-priori placeholders, "
            "which is NOT valid for evaluation."
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

    if not SOURCE_PATH.exists():
        print(f"FAILED: {SOURCE_PATH} not found (run scripts/process_data.py first)")
        return 1

    matches = pd.read_parquet(SOURCE_PATH)
    print(f"Loaded {len(matches)} matches from {SOURCE_PATH}")

    if args.estimate_through:
        try:
            elo_params = estimate_elo_params(matches, args.estimate_through)
        except ValueError as exc:
            print(f"FAILED: could not estimate Elo parameters: {exc}")
            return 1
        intended = intended_evaluation_seasons_for(args.estimate_through)
        purpose = (
            f"Fold artifact: Elo parameters estimated from seasons through "
            f"{args.estimate_through}."
        )
        valid_for_evaluation = True
        earliest_valid = _next_season(matches, args.estimate_through)
        notes = (
            f"Intended for evaluating {intended or '(no approved fold)'} only. "
            f"Call assert_artifact_valid_for() before any fit or scoring."
        )
    else:
        elo_params = canonical_elo_params()
        intended = []
        purpose = CANONICAL_PURPOSE
        valid_for_evaluation = False
        earliest_valid = None
        notes = CANONICAL_NOT_VALID_NOTE

    features = build_features(matches, elo_params)

    problems = validate_feature_frame(
        features, expected_rows=len(matches), initial_rating=elo_params.initial_rating
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
        elo_params=elo_params,
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
        notes=notes,
    )

    output_path = args.output if args.output is not None else default_output_path(
        args.estimate_through
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

    _print_report(features, elo_params, output_path, sidecar, valid_for_evaluation, intended)
    return 0


def _next_season(matches: pd.DataFrame, season: str) -> str | None:
    order = list(matches.groupby("Season")["Date"].min().sort_values().index)
    if season not in order:
        return None
    index = order.index(season) + 1
    return order[index] if index < len(order) else None


def _print_report(
    features: pd.DataFrame,
    elo_params,
    output_path: Path,
    sidecar: Path,
    valid_for_evaluation: bool,
    intended: list[str],
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

    print()
    print("Elo parameters used:")
    for key, value in elo_params_to_dict(elo_params).items():
        print(f"  {key}: {value}")

    print()
    if valid_for_evaluation:
        print(f"Artifact status: VALID for evaluating {intended or '(no approved fold)'}")
        print(f"  training cutoff: {elo_params.training_cutoff}")
    else:
        print("Artifact status: *** NOT VALID FOR MODEL EVALUATION ***")
        print("  Built with a-priori placeholder Elo parameters (no provenance).")
        print("  Use --estimate-through <fold train cutoff> to build a fold artifact.")

    print()
    cold = cold_start_row_count(features)
    print(f"Cold-start rows retained (either side < min history): {cold} ({cold / len(features):.2%})")
    print(f"Sealed season present in artifact: {SEALED_SEASON} "
          f"({int((features['Season'] == SEALED_SEASON).sum())} rows) - "
          f"never used for parameter estimation")

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
