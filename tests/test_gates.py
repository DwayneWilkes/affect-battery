"""Task 0.5 + 0.6 Red — data-collection + OSF pre-reg gates.

Per power-analysis spec "Data-collection gate" + "OSF pre-registration
top-level gate". Both gates block experiment runners before any model
calls; OSF gate fires first (pre-registration must exist before power
report can validate).
"""

import pytest

from src.gates import (
    DataCollectionGateError,
    OsfPreregistrationGateError,
    check_data_collection_gate,
    check_osf_pre_registration_gate,
)


class TestDataCollectionGate:
    def test_missing_power_report_raises(self):
        config = {
            "experiment_type": "exp1a",
            "pre_registration_osf_url": "https://osf.io/example",
            # power_report_path absent
        }
        with pytest.raises(DataCollectionGateError, match="power report"):
            check_data_collection_gate(config)

    def test_present_power_report_passes(self):
        config = {
            "experiment_type": "exp1a",
            "pre_registration_osf_url": "https://osf.io/example",
            "power_report_path": "results/power_report_2026-04-24.json",
            "power_report_sha": "0" * 64,
        }
        check_data_collection_gate(config)  # no exception

    def test_missing_power_sha_raises(self):
        config = {
            "experiment_type": "exp1a",
            "pre_registration_osf_url": "https://osf.io/example",
            "power_report_path": "results/power_report_2026-04-24.json",
            # power_report_sha absent
        }
        with pytest.raises(DataCollectionGateError, match="power"):
            check_data_collection_gate(config)


class TestOsfPreregistrationGate:
    def test_missing_osf_url_raises(self):
        config = {
            "experiment_type": "exp1a",
            # pre_registration_osf_url absent
        }
        with pytest.raises(OsfPreregistrationGateError, match="pre-registration"):
            check_osf_pre_registration_gate(config)

    def test_present_osf_url_passes(self):
        config = {
            "experiment_type": "exp1a",
            "pre_registration_osf_url": "https://osf.io/example",
            "pre_registration_sha": "a" * 64,
        }
        check_osf_pre_registration_gate(config)  # no exception

    def test_missing_pre_registration_sha_raises(self):
        config = {
            "experiment_type": "exp1a",
            "pre_registration_osf_url": "https://osf.io/example",
            # pre_registration_sha absent
        }
        with pytest.raises(OsfPreregistrationGateError, match="sha"):
            check_osf_pre_registration_gate(config)


class TestGateErrorTypes:
    def test_data_collection_error_is_value_error_subclass(self):
        assert issubclass(DataCollectionGateError, ValueError)

    def test_osf_error_is_value_error_subclass(self):
        assert issubclass(OsfPreregistrationGateError, ValueError)
