"""Pre-flight gates that block experiment runners before any model calls.

Per power-analysis spec "OSF pre-registration top-level gate" +
"Data-collection gate":

- OSF gate: pre_registration_osf_url + pre_registration_sha must be in
  the run config; the pre-reg document must exist before any
  data-collection run.
- Data-collection gate: power_report_path + power_report_sha must be in
  the run config; primary runs require a current power report.

Gates fire in order: OSF → data-collection. Each is a separate function
so callers can run them sequentially with clear error messages, and
test code can target one gate at a time.

Implementation modules:
- src/gates/osf_pre_registration.py
- src/gates/data_collection.py

`src.runner` and `src.runners.*` MUST call check_osf_pre_registration_gate
followed by check_data_collection_gate before constructing the inference
client.
"""

from .data_collection import (
    DataCollectionGateError,
    check_data_collection_gate,
)
from .osf_pre_registration import (
    OsfPreregistrationGateError,
    check_osf_pre_registration_gate,
)


__all__ = [
    "DataCollectionGateError",
    "OsfPreregistrationGateError",
    "check_data_collection_gate",
    "check_osf_pre_registration_gate",
]
