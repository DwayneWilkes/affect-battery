"""Tasks 1.3 + 1.4 + 1.5 Red — Phase 1 decision-record machinery.

Three small TDD batches:

1.3 MDE update from variance probe — power-analysis spec
"Per-hypothesis MDE coverage with grounded defaults" + design.md D3
variance-probe-override (MAX(probe_observed, default)).

1.4 Base-feasibility decision record — base-model-comparison spec
"Week-1 go/no-go gate". On fail, demote H4 base-vs-instruct to
exploratory; primary family shrinks to 4.

1.5 Budget-contingency decision record — power-analysis spec
"Budget-contingency decision record" (resolves review AI-2 / VA-M2).
On n>budget-ceiling, emit decision record naming option (a/b/c).
"""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Task 1.3 — MDE update from variance probe
# ---------------------------------------------------------------------------


class TestUpdateMdeFromProbe:
    def test_observed_below_default_keeps_default(self, tmp_path):
        from src.power.mde import update_mde_for_hypothesis

        # H1 default is 0.40 (Holm-adjusted at α=.01). Observed 0.20 → use 0.40.
        result = update_mde_for_hypothesis(
            hypothesis_id="H1",
            default_mde=0.40,
            observed_effect_size=0.20,
        )
        assert result["mde_used"] == 0.40
        assert result["mde_source"] == "default"

    def test_observed_above_default_uses_observed(self, tmp_path):
        from src.power.mde import update_mde_for_hypothesis

        result = update_mde_for_hypothesis(
            hypothesis_id="H3a",
            default_mde=0.08,
            observed_effect_size=0.12,
        )
        assert result["mde_used"] == 0.12
        assert result["mde_source"] == "pilot_observed"

    def test_no_observed_uses_default(self):
        from src.power.mde import update_mde_for_hypothesis

        result = update_mde_for_hypothesis(
            hypothesis_id="H1",
            default_mde=0.40,
            observed_effect_size=None,
        )
        assert result["mde_used"] == 0.40
        assert result["mde_source"] == "default"


# ---------------------------------------------------------------------------
# Task 1.4 — base-feasibility decision record
# ---------------------------------------------------------------------------


class TestBaseFeasibilityDecision:
    def test_pass_verdict_keeps_h4_primary(self, tmp_path):
        from src.gates.base_feasibility_decision import apply_decision

        amendment = apply_decision(
            probe_verdict="pass",
            output_dir=tmp_path,
        )
        assert amendment["h4_status"] == "primary"
        assert amendment["primary_family_size"] == 5

    def test_fail_verdict_demotes_h4(self, tmp_path):
        from src.gates.base_feasibility_decision import apply_decision

        amendment = apply_decision(
            probe_verdict="fail",
            output_dir=tmp_path,
        )
        assert amendment["h4_status"] == "exploratory"
        assert amendment["primary_family_size"] == 4

    def test_writes_amendment_record(self, tmp_path):
        from src.gates.base_feasibility_decision import apply_decision
        apply_decision(probe_verdict="fail", output_dir=tmp_path)
        files = list(tmp_path.glob("base_feasibility_decision_*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record["h4_status"] == "exploratory"


# ---------------------------------------------------------------------------
# Task 1.5 — budget-contingency decision record
# ---------------------------------------------------------------------------


class TestBudgetContingencyDecision:
    def test_within_budget_emits_no_decision(self, tmp_path):
        from src.power.budget_contingency import emit_decision

        decision = emit_decision(
            recommended_n_per_condition=50,
            budget_ceiling_n=200,  # within budget
            output_dir=tmp_path,
        )
        assert decision is None  # nothing to do

    def test_over_budget_emits_decision_record(self, tmp_path):
        from src.power.budget_contingency import emit_decision

        decision = emit_decision(
            recommended_n_per_condition=300,
            budget_ceiling_n=200,
            output_dir=tmp_path,
            chosen_option="b",  # accept lower power on H3b/H3c
        )
        assert decision is not None
        assert decision["chosen_option"] == "b"
        files = list(tmp_path.glob("budget_contingency_*.yaml"))
        assert len(files) == 1

    def test_invalid_option_rejected(self, tmp_path):
        from src.power.budget_contingency import emit_decision

        with pytest.raises(ValueError, match="option"):
            emit_decision(
                recommended_n_per_condition=300,
                budget_ceiling_n=200,
                output_dir=tmp_path,
                chosen_option="z",
            )

    def test_option_b_records_section_6_commitment(self, tmp_path):
        """Option (b) MUST cite paper §6 reporting commitment per spec."""
        from src.power.budget_contingency import emit_decision
        decision = emit_decision(
            recommended_n_per_condition=300,
            budget_ceiling_n=200,
            output_dir=tmp_path,
            chosen_option="b",
        )
        assert "§6" in decision["rationale"]
