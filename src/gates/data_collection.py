"""Data-collection gate (power-analysis spec)."""

from __future__ import annotations


class DataCollectionGateError(ValueError):
    """Raised when a primary data-collection run is invoked without a
    current power report on file."""


def check_data_collection_gate(config: dict) -> None:
    """Verify the run config carries a current power report reference.

    Per power-analysis spec "Data-collection gate": no Exp 1a/1b/2/3a/3b/3c
    primary run advances without:

    - `power_report_path` (path to the per-experiment power report JSON)
    - `power_report_sha` (SHA-256 of the canonicalized report)

    Raises:
        DataCollectionGateError: if either field is missing or empty.
    """
    path = config.get("power_report_path")
    if not path:
        raise DataCollectionGateError(
            "Run blocked: power report missing. "
            "Config must include `power_report_path` referencing the "
            "current per-experiment power report. See paper §3.1 + "
            "power-analysis spec 'Data-collection gate'."
        )
    sha = config.get("power_report_sha")
    if not sha:
        raise DataCollectionGateError(
            "Run blocked: power report sha missing. "
            "Config must include `power_report_sha`."
        )
