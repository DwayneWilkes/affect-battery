"""OSF pre-registration finalization (Tasks 2.0a + 2.1 + 2.2)."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.power.mde import update_mde_for_hypothesis


def _canonical_yaml_bytes(data: dict[str, Any]) -> bytes:
    """Serialize YAML with stable key order + no end-of-doc spaces.

    Cross-runtime-stable so SHA-256 doesn't drift across machines.
    """
    return yaml.safe_dump(
        data,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    ).encode("utf-8")


def compute_prereg_sha(yaml_path: Path) -> str:
    """Compute SHA-256 of the canonicalized YAML at `yaml_path`.

    Excludes the `pre_registration_sha` field itself from the hash
    (otherwise self-reference would break determinism).
    """
    data = yaml.safe_load(Path(yaml_path).read_text())
    data = {k: v for k, v in data.items() if k != "pre_registration_sha"}
    return hashlib.sha256(_canonical_yaml_bytes(data)).hexdigest()


def _set_sha_in_place(yaml_path: Path) -> str:
    """Compute SHA, write it into the file's pre_registration_sha
    field, return the SHA."""
    sha = compute_prereg_sha(yaml_path)
    data = yaml.safe_load(yaml_path.read_text())
    data["pre_registration_sha"] = sha
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=True))
    return sha


def finalize_v1(
    v0_path: Path,
    output_path: Path,
    observed_effect_sizes: dict[str, float],
    base_feasibility_verdict: str,
    rationale: str = "v0 → v1: probe-grounded MDEs + base-feasibility decision",
) -> None:
    """Read v0 YAML, apply probe-grounded MDE updates + base-feasibility
    decision, append amendment_chain entry, write v1 with computed SHA.

    Args:
        v0_path: input YAML.
        output_path: where to write v1 (may equal v0_path).
        observed_effect_sizes: hypothesis_id → observed effect size from probe.
        base_feasibility_verdict: "pass" | "fail" from BaseModelProbeResult.
        rationale: amendment_chain entry rationale.
    """
    v0_path = Path(v0_path)
    output_path = Path(output_path)
    data = yaml.safe_load(v0_path.read_text())

    # Apply probe-grounded MDEs per src.power.mde rule
    for h in data.get("hypotheses", []):
        observed = observed_effect_sizes.get(h["id"])
        update = update_mde_for_hypothesis(
            hypothesis_id=h["id"],
            default_mde=h["mde"],
            observed_effect_size=observed,
        )
        h["mde"] = update["mde_used"]
        h["mde_source"] = update["mde_source"]

    # Apply base-feasibility decision
    if base_feasibility_verdict == "fail":
        for h in data.get("hypotheses", []):
            if h["id"] == "H4":
                h["correction_family"] = "exploratory"
                h.setdefault("notes", "")
                h["notes"] += " Demoted from primary per base-feasibility decision."
        cf = data.get("correction_families", {})
        if "H4" in cf.get("primary", []):
            cf["primary"] = [x for x in cf["primary"] if x != "H4"]
            cf.setdefault("exploratory", []).append("H4")

    # Append amendment_chain entry
    amendment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rationale": rationale,
        "sha": "",  # filled in below
    }
    data.setdefault("amendment_chain", []).append(amendment)

    # Write + compute SHA + write again (SHA-of-payload-without-SHA)
    output_path.write_text(yaml.safe_dump(data, sort_keys=True))
    sha = _set_sha_in_place(output_path)

    # Record SHA in the amendment we just appended
    data = yaml.safe_load(output_path.read_text())
    data["amendment_chain"][-1]["sha"] = sha
    output_path.write_text(yaml.safe_dump(data, sort_keys=True))


def prepare_for_submit(yaml_path: Path, osf_url: str) -> str:
    """Record the OSF URL after upload + recompute SHA.

    Returns the final SHA written to the YAML.
    """
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text())
    data["osf_url"] = osf_url
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=True))
    return _set_sha_in_place(yaml_path)
