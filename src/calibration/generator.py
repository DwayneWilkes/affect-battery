#!/usr/bin/env python3
"""Generate candidate items for `configs/banks/arithmetic_hard_v1.yaml`.

Produces items per the design.md D1 schema:
    id, operands, operator, answer, digit_count, n_carries

n_carries is the total count of carry or borrow operations a human would
perform executing the standard pen-and-paper algorithm. Semantics:
    add: column carries
    sub: column borrows
    mul: partial-product column carries + carries in the partial-product sum
    div: sum of mul carries and sub borrows across long-division steps

Per-operator item generation is parameterized so the auto-calibrator
(`scripts/auto_calibrate_arithmetic.py`) can probe difficulty by varying
digit_level. Module-level defaults below reproduce the hand-tuned bank.

Seeded for determinism; re-runs with the same seed are byte-identical.
"""

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import yaml


BANK_ID = "arithmetic_hard_v1"
BANK_VERSION = 1
TOTAL_ITEMS = 300
RNG_SEED = 20260421

OPERATOR_MIX = OrderedDict([
    ("add", 0.50),
    ("sub", 0.30),
    ("mul", 0.15),
    ("div", 0.05),
])


# ───────────────────────── carry / borrow counters ──────────────────────────


def count_add_carries(a: int, b: int) -> int:
    """Number of carry operations in column-wise addition of a + b."""
    carries = 0
    carry = 0
    while a > 0 or b > 0 or carry > 0:
        s = (a % 10) + (b % 10) + carry
        carry = 1 if s >= 10 else 0
        if carry:
            carries += 1
        a //= 10
        b //= 10
    return carries


def count_sub_borrows(a: int, b: int) -> int:
    """Number of borrow operations in a - b, assuming a >= b."""
    borrows = 0
    borrow = 0
    while a > 0 or b > 0:
        da = (a % 10) - borrow
        db = b % 10
        if da < db:
            borrow = 1
            borrows += 1
        else:
            borrow = 0
        a //= 10
        b //= 10
    return borrows


def count_mul_long_carries(a: int, b: int) -> int:
    """Count actual carries in long multiplication of a * b.

    Partial products (single-digit × multi-digit, with column carries) plus
    carries in the final sum of partial products.
    """
    carries = 0
    b_digits = []
    temp = b
    while temp > 0:
        b_digits.append(temp % 10)
        temp //= 10
    if not b_digits:
        return 0

    partials: list[int] = []
    for shift, d in enumerate(b_digits):
        if d == 0:
            partials.append(0)
            continue
        carry = 0
        product = 0
        multiplier = 1
        temp_a = a
        while temp_a > 0 or carry > 0:
            s = (temp_a % 10) * d + carry
            product += (s % 10) * multiplier
            new_carry = s // 10
            if new_carry > 0:
                carries += 1
            carry = new_carry
            temp_a //= 10
            multiplier *= 10
        partials.append(product * (10 ** shift))

    total = 0
    for p in partials:
        if p == 0:
            continue
        if total == 0:
            total = p
        else:
            carries += count_add_carries(total, p)
            total += p

    return carries


def count_div_long_carries(a: int, b: int) -> int:
    """Count carries in long division of a / b (sum of mul carries and sub
    borrows across every long-division step)."""
    if b == 0 or a < b:
        return 0

    carries = 0
    partial = 0
    for digit in str(a):
        partial = partial * 10 + int(digit)
        if partial >= b:
            q_digit = partial // b
            carries += count_mul_long_carries(b, q_digit)
            subtract_value = b * q_digit
            carries += count_sub_borrows(partial, subtract_value)
            partial -= subtract_value
    return carries


# ───────────────────────── item-level helpers ──────────────────────────


def digit_count(*operands: int) -> list[int]:
    """Per-operand digit counts aligned with operand order."""
    return [len(str(op)) for op in operands]


def _int_range_for_digits(d: int) -> tuple[int, int]:
    """Return (lo, hi) int range for an integer of exactly d digits.

    d=1 → (1, 9); d=2 → (10, 99); etc. Note: we exclude 0 to keep operands
    non-trivially sized at all digit levels.
    """
    assert d >= 1
    lo = 10 ** (d - 1) if d > 1 else 1
    hi = 10 ** d - 1
    return lo, hi


# ───────────────────────── per-operator generators ──────────────────────────


@dataclass
class GenSpec:
    """Difficulty parameters for a single-item generator call.

    digit_level is the primary knob used by the auto-calibrator. Each
    generator interprets it operator-specifically:
        add/sub: both operands at exactly digit_level digits
        mul: operand_a at digit_level, operand_b at max(2, digit_level-1) digits
             (mul is asymmetric to keep difficulty in-window without hitting floor)
        div: divisor and quotient each at digit_level, dividend derived
    n_carries_range is a constraint filter applied after generation.
    """
    digit_level: int
    n_carries_range: tuple[int, int] = (1, 999)


def gen_add(rng: random.Random, spec: GenSpec) -> dict | None:
    lo, hi = _int_range_for_digits(spec.digit_level)
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    carries = count_add_carries(a, b)
    nc_lo, nc_hi = spec.n_carries_range
    if not (nc_lo <= carries <= nc_hi):
        return None
    return {
        "operands": [a, b],
        "operator": "add",
        "answer": a + b,
        "digit_count": digit_count(a, b),
        "n_carries": carries,
    }


def gen_sub(rng: random.Random, spec: GenSpec) -> dict | None:
    """Subtraction with a >= b so the answer is non-negative."""
    lo, hi = _int_range_for_digits(spec.digit_level)
    a = rng.randint(lo, hi)
    b = rng.randint(lo, min(a, hi))
    borrows = count_sub_borrows(a, b)
    nc_lo, nc_hi = spec.n_carries_range
    if not (nc_lo <= borrows <= nc_hi):
        return None
    return {
        "operands": [a, b],
        "operator": "sub",
        "answer": a - b,
        "digit_count": digit_count(a, b),
        "n_carries": borrows,
    }


def gen_mul(rng: random.Random, spec: GenSpec) -> dict | None:
    """Multiplication: asymmetric operand sizing.

    Keeps difficulty in-window: 3×3 digit multiplication hits the floor on 7B
    models; we size the smaller operand at max(2, digit_level-1) so the item
    remains solvable while still exercising the target digit_level.
    """
    lo_a, hi_a = _int_range_for_digits(spec.digit_level)
    smaller_digits = max(2, spec.digit_level - 1)
    lo_b, hi_b = _int_range_for_digits(smaller_digits)
    a = rng.randint(lo_a, hi_a)
    b = rng.randint(lo_b, hi_b)
    carries = count_mul_long_carries(a, b)
    nc_lo, nc_hi = spec.n_carries_range
    if not (nc_lo <= carries <= nc_hi):
        return None
    return {
        "operands": [a, b],
        "operator": "mul",
        "answer": a * b,
        "digit_count": digit_count(a, b),
        "n_carries": carries,
    }


def gen_div(rng: random.Random, spec: GenSpec) -> dict | None:
    """Exact integer division. Picks divisor and quotient at digit_level
    digits, sets dividend = divisor * quotient so the answer is always an
    integer."""
    lo, hi = _int_range_for_digits(spec.digit_level)
    divisor = rng.randint(lo, hi)
    quotient = rng.randint(lo, hi)
    dividend = divisor * quotient
    carries = count_div_long_carries(dividend, divisor)
    nc_lo, nc_hi = spec.n_carries_range
    if not (nc_lo <= carries <= nc_hi):
        return None
    return {
        "operands": [dividend, divisor],
        "operator": "div",
        "answer": quotient,
        "digit_count": digit_count(dividend, divisor),
        "n_carries": carries,
    }


GENERATORS = {
    "add": gen_add,
    "sub": gen_sub,
    "mul": gen_mul,
    "div": gen_div,
}


# Per-operator default specs that reproduce the 2026-04-21 hand-tuned bank.
# The auto-calibrator overrides these with empirically-derived specs.
DEFAULT_OPERATOR_SPECS: dict[str, GenSpec] = {
    "add": GenSpec(digit_level=4, n_carries_range=(1, 3)),
    "sub": GenSpec(digit_level=4, n_carries_range=(1, 3)),
    "mul": GenSpec(digit_level=3, n_carries_range=(1, 8)),
    "div": GenSpec(digit_level=2, n_carries_range=(1, 8)),
}


# ───────────────────────── bank assembly ──────────────────────────


def generate_items_for_operator(
    operator: str,
    spec: GenSpec,
    count: int,
    rng: random.Random,
    id_prefix: str = "item",
    start_idx: int = 1,
    max_attempts_per_item: int = 200,
) -> list[dict]:
    """Generate `count` items for the given operator under `spec`.

    Raises RuntimeError if the difficulty constraint is so tight the
    generator can't find items in max_attempts_per_item tries on average.
    """
    gen = GENERATORS[operator]
    items: list[dict] = []
    attempts = 0
    max_attempts = count * max_attempts_per_item
    while len(items) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Generator for {operator} at {spec} produced only "
                f"{len(items)}/{count} items after {attempts} attempts; "
                f"difficulty constraints may be infeasible."
            )
        candidate = gen(rng, spec)
        if candidate is None:
            continue
        candidate["id"] = f"{id_prefix}_{start_idx + len(items):04d}"
        items.append(candidate)
    return items


def generate_items(
    total: int = TOTAL_ITEMS,
    operator_mix: dict[str, float] = None,
    operator_specs: dict[str, GenSpec] = None,
    rng_seed: int = RNG_SEED,
    id_prefix: str = "arith_hard_v1",
) -> list[dict]:
    """Generate a full item set honoring operator_mix proportions."""
    operator_mix = operator_mix or OPERATOR_MIX
    operator_specs = operator_specs or DEFAULT_OPERATOR_SPECS
    rng = random.Random(rng_seed)

    target_counts = {op: round(total * frac) for op, frac in operator_mix.items()}
    delta = total - sum(target_counts.values())
    first_op = next(iter(operator_mix))
    target_counts[first_op] += delta

    items: list[dict] = []
    idx = 1
    for op in operator_mix:
        target = target_counts[op]
        bucket = generate_items_for_operator(
            operator=op,
            spec=operator_specs[op],
            count=target,
            rng=rng,
            id_prefix=id_prefix,
            start_idx=idx,
        )
        items.extend(bucket)
        idx += len(bucket)
    return items


def reorder_item(item: dict) -> OrderedDict:
    """Emit keys in a stable order for human-readable YAML diffs."""
    return OrderedDict([
        ("id", item["id"]),
        ("operands", item["operands"]),
        ("operator", item["operator"]),
        ("answer", item["answer"]),
        ("digit_count", item["digit_count"]),
        ("n_carries", item["n_carries"]),
    ])


def build_bank_yaml(
    items: list[dict],
    bank_id: str = BANK_ID,
    bank_version: int = BANK_VERSION,
    expected_accuracy_class: str = "mid",
    operator_mix: dict[str, float] = None,
    operator_specs: dict[str, GenSpec] = None,
    calibration_source: str | None = None,
) -> OrderedDict:
    """Assemble the bank YAML with a calibration-source annotation so readers
    know whether the bank was hand-tuned or auto-calibrated."""
    operator_mix = operator_mix or OPERATOR_MIX
    operator_specs = operator_specs or DEFAULT_OPERATOR_SPECS

    all_digits = [d for item in items for d in item["digit_count"]]
    observed_digit_range = [min(all_digits), max(all_digits)]

    difficulty_profile = OrderedDict([
        ("expected_accuracy_class", expected_accuracy_class),
        ("per_operand_digit_range", observed_digit_range),
        ("operator_mix", dict(operator_mix)),
        ("operator_specs", {
            op: {
                "digit_level": spec.digit_level,
                "n_carries_range": list(spec.n_carries_range),
            }
            for op, spec in operator_specs.items()
        }),
        ("calibration_source", calibration_source or "hand-tuned defaults"),
        ("notes",
         "digit_count is a list of per-operand digit counts aligned with "
         "the operands list (e.g., operands=[210, 72] => digit_count=[3, 2]). "
         "n_carries is the total count of carry or borrow operations a human "
         "executing the standard pen-and-paper algorithm would perform. "
         "Per-operator semantics: add=column carries, sub=column borrows, "
         "mul=partial-product carries plus final-sum carries, "
         "div=sum of mul carries and sub borrows across long-division steps."),
    ])

    return OrderedDict([
        ("bank_id", bank_id),
        ("bank_version", bank_version),
        ("difficulty_profile", difficulty_profile),
        ("items", [reorder_item(item) for item in items]),
    ])


# Represent OrderedDict as a plain YAML mapping (preserves key order).
def _ordered_dict_representer(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, _ordered_dict_representer)


def write_bank(bank: OrderedDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.dump(bank, f, sort_keys=False, default_flow_style=False, width=120)


def summarize(items: list[dict]) -> None:
    by_op: dict[str, list[dict]] = {}
    for item in items:
        by_op.setdefault(item["operator"], []).append(item)
    print(f"Generated {len(items)} items.")
    for op in ("add", "sub", "mul", "div"):
        bucket = by_op.get(op, [])
        if not bucket:
            continue
        all_digits = [d for item in bucket for d in item["digit_count"]]
        ncs = [i["n_carries"] for i in bucket]
        print(
            f"  {op:>3}: {len(bucket):>3} items  "
            f"per-operand digits min/max = {min(all_digits)}/{max(all_digits)}  "
            f"n_carries min/max = {min(ncs)}/{max(ncs)}"
        )


def main() -> None:
    items = generate_items()
    summarize(items)
    bank = build_bank_yaml(items)
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "configs" / "banks" / f"{BANK_ID}.yaml"
    write_bank(bank, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
