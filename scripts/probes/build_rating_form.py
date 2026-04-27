"""Generate per-rater intensity rating forms.

Each rater receives a YAML form containing the 7 INTENSITY_LEVELS stimuli
in randomized order (controlling for order-effect bias). The rater reads
each stimulus and writes an integer rating 1-7 in the `rating` field.
The intensity scale is signed:

    1 = strongest positive
    4 = neutral / no affect
    7 = strongest negative

After raters fill out their forms, run scripts/probes/run_intensity_pilot.py
to compute Krippendorff α and emit a pilot seed if the gate passes.

Usage:
    uv run python scripts/probes/build_rating_form.py \\
        --rater-id rater_A \\
        --output ratings/rater_A.yaml \\
        --seed 1
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow importing src.* without packaging the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.conditioning.prompts import INTENSITY_LEVELS  # noqa: E402


FORM_HEADER = """# Intensity-axis rating form
#
# Instructions:
#   1. Confirm `rater_id` below matches the pseudonymous ID you were
#      assigned (e.g. rater_A, rater_B, rater_PI). Do not enter
#      personally identifying information.
#   2. For each stimulus, read the `feedback_text` carefully.
#   3. Assign an integer rating 1-7 in the `rating` field, where:
#         1 = strongest positive feedback
#         4 = neutral / no affective content
#         7 = strongest negative feedback
#   4. Do NOT change the order of stimuli or the `id` values.
#   5. Save and return this file to the operator.
#
# Stimuli are presented below in a randomized order to control for
# order-effect bias. Each rater's form has a different randomization.
"""


def build_form(rater_id: str, seed: int) -> str:
    """Render a YAML form template with stimuli in seed-randomized order."""
    rng = random.Random(seed)
    stimuli = list(INTENSITY_LEVELS)
    rng.shuffle(stimuli)

    lines = [FORM_HEADER, ""]
    lines.append(f"rater_id: {rater_id}")
    lines.append(f"randomization_seed: {seed}")
    lines.append("")
    lines.append("ratings:")
    for stim in stimuli:
        lines.append(f"  - id: level_{stim.level}")
        # Quote the feedback text to keep YAML safe; preserve any internal quotes.
        safe_text = stim.feedback_text.replace('"', '\\"')
        lines.append(f"    feedback_text: \"{safe_text}\"")
        lines.append(f"    rating: # FILL IN: integer 1-7")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rater-id", required=True,
                    help="Pseudonymous rater identifier (e.g. rater_A, "
                         "rater_PI). Stored verbatim in the form and the "
                         "downstream seed JSON; do not pass personally "
                         "identifying information.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write the form YAML.")
    ap.add_argument("--seed", type=int, required=True,
                    help="Randomization seed for stimulus order. Use a "
                         "different seed per rater so each rater sees a "
                         "different ordering.")
    args = ap.parse_args()

    form_text = build_form(args.rater_id, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(form_text, encoding="utf-8")
    print(f"Form written to {args.output}", file=sys.stderr)
    print(f"  rater_id: {args.rater_id}", file=sys.stderr)
    print(f"  randomization_seed: {args.seed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
