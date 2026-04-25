"""Task 8.2 Red — base-vs-instruct asymmetry contrast.

Per asymmetry-contrast spec "Base-vs-instruct asymmetry contrast" +
"Both primary tests pre-registered": for the Llama-3-8B family,
asymmetry_delta_ratio = ratio_instruct / ratio_base. Both pre-registered
tests run:
  (a) primary: delta_ratio > 1
  (b) secondary: difference_instruct > difference_base
"""

from __future__ import annotations

import pytest


class TestBaseVsInstructContrast:
    def test_base_vs_instruct_contrast(self):
        from src.analysis.asymmetry import contrast_base_vs_instruct

        per_model = {
            "Meta-Llama-3-8B": {
                "ratio_geomean": 1.5,
                "difference_mean": 0.10,
            },
            "Meta-Llama-3-8B-Instruct": {
                "ratio_geomean": 3.0,
                "difference_mean": 0.25,
            },
        }
        result = contrast_base_vs_instruct(
            per_model,
            base_model="Meta-Llama-3-8B",
            instruct_model="Meta-Llama-3-8B-Instruct",
        )
        # delta_ratio = 3.0 / 1.5 = 2.0
        assert result["asymmetry_delta_ratio"] == 2.0
        # Both pre-registered tests run
        assert "test_a_primary_delta_ratio_gt_1" in result
        assert "test_b_secondary_diff_instruct_gt_diff_base" in result
        assert result["test_a_primary_delta_ratio_gt_1"] is True
        assert result["test_b_secondary_diff_instruct_gt_diff_base"] is True

    def test_missing_model_raises(self):
        from src.analysis.asymmetry import contrast_base_vs_instruct
        with pytest.raises(ValueError, match="missing"):
            contrast_base_vs_instruct({}, base_model="X", instruct_model="Y")
