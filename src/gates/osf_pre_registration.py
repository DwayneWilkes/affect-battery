"""OSF pre-registration top-level gate (power-analysis spec)."""

from __future__ import annotations


class OsfPreregistrationGateError(ValueError):
    """Raised when a run is invoked without a valid OSF pre-registration
    URL + content-hash in its config."""


def check_osf_pre_registration_gate(config: dict) -> None:
    """Verify the run config carries OSF pre-registration metadata.

    Per power-analysis spec "OSF pre-registration top-level gate":
    - `pre_registration_osf_url` MUST be present (the OSF page URL).
    - `pre_registration_sha` MUST be present (SHA-256 of the canonicalized
      pre-registration YAML; cross-validated against the power report's
      same field).

    Raises:
        OsfPreregistrationGateError: if either field is missing or empty.
    """
    osf_url = config.get("pre_registration_osf_url")
    if not osf_url:
        raise OsfPreregistrationGateError(
            "Run blocked: pre-registration missing. "
            "Config must include `pre_registration_osf_url` referencing "
            "the OSF-hosted pre-registration document. See paper §6 + "
            "power-analysis spec 'OSF pre-registration top-level gate'."
        )
    sha = config.get("pre_registration_sha")
    if not sha:
        raise OsfPreregistrationGateError(
            "Run blocked: pre-registration sha missing. "
            "Config must include `pre_registration_sha` (SHA-256 of the "
            "canonicalized pre-registration YAML). The power report's "
            "pre_registration_sha field MUST match this value."
        )
