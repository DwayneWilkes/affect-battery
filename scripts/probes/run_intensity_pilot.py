"""Drive the intensity-axis pilot: load filled rating forms, compute
Krippendorff α, emit a signed pilot seed if the gate passes.

Inputs:
    A directory of completed rater YAML forms (one per rater, ≥3 raters).
    Each form must follow the schema produced by build_rating_form.py and
    must have all `rating` fields filled with integers 1-7.

Output:
    Stdout: human-readable α report + pilot decision.
    On 'proceed' decision: a signed pilot-seed JSON written to the
    --output path (default: configs/intensity_pilot_seed.json) for use as
    the H3a runner's pilot_seed_path.

Decision rule (from src/probes/intensity_pilot.py):
    α >= 0.8 overall AND all adjacent pairs >= 0.6 -> proceed
    α < 0.6 overall OR any adjacent pair < 0.6 -> collapse
    Otherwise -> restructure (refine prompts and re-run)

Usage:
    uv run python scripts/probes/run_intensity_pilot.py \\
        --ratings-dir ratings/ \\
        --output configs/intensity_pilot_seed.json \\
        --pilot-date 2026-04-27 \\
        --axis-id intensity_axis_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import random  # noqa: E402

from src.conditioning.prompts import INTENSITY_LEVELS  # noqa: E402
from src.probes.intensity_pilot import (  # noqa: E402
    emit_seed,
    emit_solo_seed,
    run_intensity_pilot,
)


def _opaque_to_level_mapping(seed: int) -> dict[str, int]:
    """Re-derive the opaque-id → canonical-level mapping from a seed.

    Mirrors `build_rating_form.py::build_form`'s shuffling: the same RNG
    seed produces the same presentation order, and presentation position
    i (1-indexed) maps to opaque id `stim_{i:03d}`.
    """
    rng = random.Random(seed)
    stimuli = list(INTENSITY_LEVELS)
    rng.shuffle(stimuli)
    return {
        f"stim_{position:03d}": stim.level
        for position, stim in enumerate(stimuli, start=1)
    }


def load_rater_form(path: Path) -> tuple[str, list[int]]:
    """Parse a filled rater form. Returns (rater_id, ratings_in_canonical_level_order).

    The form's stimulus identifiers are opaque (`stim_001` through
    `stim_007`); we re-derive the opaque-id → canonical-level mapping
    from the form's `randomization_seed` and re-sort ratings into
    canonical level order [level_1, ..., level_7]. This keeps the
    (rater × item) matrix aligned for Krippendorff α and prevents
    rater-side inference of the canonical ordering from the form.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rater_id = data.get("rater_id")
    if not isinstance(rater_id, str) or not rater_id.strip():
        raise ValueError(f"{path}: missing or invalid rater_id")
    seed = data.get("randomization_seed")
    if not isinstance(seed, int):
        raise ValueError(
            f"{path}: missing or non-integer randomization_seed (got {seed!r})"
        )
    raw_ratings = data.get("ratings")
    if not isinstance(raw_ratings, list):
        raise ValueError(f"{path}: ratings field must be a list")

    opaque_to_level = _opaque_to_level_mapping(seed)

    by_level: dict[int, int] = {}
    for entry in raw_ratings:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: ratings entries must be mappings")
        opaque_id = entry.get("id")
        rating = entry.get("rating")
        if opaque_id is None:
            raise ValueError(f"{path}: ratings entry missing 'id'")
        if opaque_id not in opaque_to_level:
            raise ValueError(
                f"{path}: unknown stimulus id {opaque_id!r}; expected one of "
                f"{sorted(opaque_to_level)}. Form may have been generated with "
                f"a different seed than recorded."
            )
        if rating is None or not isinstance(rating, int):
            raise ValueError(
                f"{path}: rating for {opaque_id!r} is missing or not an "
                f"integer (got {rating!r})"
            )
        if not (1 <= rating <= 7):
            raise ValueError(
                f"{path}: rating for {opaque_id!r} out of range "
                f"(got {rating}; must be 1-7)"
            )
        level = opaque_to_level[opaque_id]
        by_level[level] = rating

    expected_levels = set(range(1, 8))
    missing = expected_levels - set(by_level.keys())
    if missing:
        raise ValueError(
            f"{path}: missing ratings for canonical levels {sorted(missing)}"
        )

    canonical = [by_level[i] for i in range(1, 8)]
    return rater_id, canonical


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings-dir", required=True, type=Path,
                    help="Directory containing one filled YAML form per rater")
    ap.add_argument("--output", type=Path,
                    default=Path("configs/intensity_pilot_seed.json"),
                    help="Path to write the pilot-seed artifact (only on "
                         "'proceed' decision)")
    ap.add_argument("--pilot-date", required=True,
                    help="Pilot date as YYYY-MM-DD; recorded in the seed")
    ap.add_argument("--axis-id", default="intensity_axis_v1",
                    help="Axis identifier; recorded in the seed")
    ap.add_argument("--solo-rater", action="store_true",
                    help="Single-rater pilot. Accepts one form and emits a "
                         "seed marked irr_validated=false without computing "
                         "Krippendorff α. Downstream gates read the flag.")
    args = ap.parse_args()

    if not args.ratings_dir.is_dir():
        print(f"ratings-dir does not exist: {args.ratings_dir}", file=sys.stderr)
        return 2

    forms = sorted(args.ratings_dir.glob("*.yaml")) + sorted(
        args.ratings_dir.glob("*.yml")
    )
    min_required = 1 if args.solo_rater else 3
    if len(forms) < min_required:
        print(f"need at least {min_required} rater form{'s' if min_required > 1 else ''}; "
              f"found {len(forms)} in {args.ratings_dir}", file=sys.stderr)
        return 2

    ratings: dict[str, list[int]] = {}
    for form_path in forms:
        try:
            rater_id, canonical = load_rater_form(form_path)
        except (ValueError, yaml.YAMLError) as e:
            print(f"FAIL parsing {form_path}: {e}", file=sys.stderr)
            return 1
        if rater_id in ratings:
            print(f"duplicate rater_id {rater_id!r} (in {form_path})",
                  file=sys.stderr)
            return 1
        ratings[rater_id] = canonical

    # Single-rater path: emit a seed without computing α.
    if args.solo_rater:
        if len(ratings) != 1:
            print(f"--solo-rater requires exactly 1 form; found {len(ratings)}",
                  file=sys.stderr)
            return 2
        rater_id, canonical = next(iter(ratings.items()))
        print("=" * 50)
        print("  Single-rater pilot")
        print("=" * 50)
        print(f"  Rater:        {rater_id}")
        print(f"  Items:        {len(canonical)}")
        print(f"  Ratings:      {canonical}")
        print()
        print("  Krippendorff α is not computed for single-rater pilots.")
        print("  The seed is marked irr_validated=false; downstream gates")
        print("  read the flag and apply their own acceptance policy.")
        print()
        seed_path = emit_solo_seed(
            rater_id=rater_id,
            ratings=canonical,
            axis_id=args.axis_id,
            n_levels=7,
            pilot_date=args.pilot_date,
            output_path=args.output,
        )
        print(f"Single-rater seed written to {seed_path}.")
        return 0

    # Multi-rater path: compute Krippendorff α + decision.
    result = run_intensity_pilot(ratings)

    print("=" * 50)
    print("  Intensity-axis pilot result")
    print("=" * 50)
    print(f"  Raters:        {result['n_raters']} ({', '.join(result['raters'])})")
    print(f"  Items:         {result['n_items']} (the 7 INTENSITY_LEVELS)")
    print(f"  α overall:     {result['alpha_overall']:.3f}")
    print(f"  α pairwise:")
    for pair, alpha in sorted(result["alpha_pairwise"].items()):
        print(f"    {pair}: {alpha:.3f}")
    print(f"  Decision:      {result['decision']}")
    print()
    print(f"  Per the gate (src/probes/intensity_pilot.py):")
    print(f"    'proceed'    requires α ≥ 0.8 overall AND all pairs ≥ 0.6")
    print(f"    'collapse'   triggers when α < 0.6 overall OR any pair < 0.6")
    print(f"    'restructure' triggers in between")
    print()

    if result["decision"] != "proceed":
        print(f"Pilot did NOT pass; no seed written.")
        if result["decision"] == "collapse":
            print(f"Recommended response: collapse adjacent levels or "
                  f"restructure to a single-mechanism axis.")
        else:
            print(f"Recommended response: refine the stimulus prompts and "
                  f"re-run the pilot.")
        return 1

    seed_path = emit_seed(
        pilot_result=result,
        axis_id=args.axis_id,
        n_levels=7,
        pilot_date=args.pilot_date,
        output_path=args.output,
    )
    print(f"Pilot PASSED. Seed artifact written to {seed_path}.")
    print(f"This file is the input to the H3a runner's --pilot-seed-path flag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
