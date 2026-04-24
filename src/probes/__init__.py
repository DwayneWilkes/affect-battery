"""Week-0 probes (design.md Phase 1).

Two probes ground the OSF pre-registration before primary data collection:

- variance probe: small-n manipulation-check on a single instruct model;
  yields variance_estimate + observed_effect_size for MDE updates
  (consumed by Task 1.3 update logic).
- base-model feasibility probe: 5 GSM8K problems on Llama-3-8B base;
  passes if baseline accuracy ≥ 0.30 per
  base-model-comparison spec "Week-1 go/no-go gate for base-model feasibility".
"""
