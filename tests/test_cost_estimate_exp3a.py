"""Cost estimator for the single-turn exp3a paradigm.

Each cell is one short two-message exchange (system + user); the
estimator must NOT include a multi-turn conditioning prefix in the
per-call input token count.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.cli import _estimate_call_token_sequence, _TOKENS_PER_TURN


def _exp3a_args(intensity_levels=None):
    return SimpleNamespace(
        experiment="exp3a",
        num_conditioning_turns=5,
        num_transfer_questions=5,
        neutral_turns=0,
    )


def test_exp3a_estimate_returns_one_call_per_level():
    args = _exp3a_args()
    levels = [1, 2, 3, 4, 5, 6, 7]
    calls = _estimate_call_token_sequence(args, {"intensity_levels": levels})
    assert len(calls) == 7


def test_exp3a_estimate_has_no_conditioning_prefix():
    """Per-call input is just sys_tok + tq, not the multi-turn accumulator."""
    args = _exp3a_args()
    calls = _estimate_call_token_sequence(args, {"intensity_levels": [1, 2, 3, 4, 5, 6, 7]})
    expected_input = _TOKENS_PER_TURN["system_prompt"] + _TOKENS_PER_TURN["transfer_q"]
    expected_output = _TOKENS_PER_TURN["transfer_a_qa"]
    for in_tok, out_tok in calls:
        assert in_tok == expected_input
        assert out_tok == expected_output


def test_exp3a_per_call_input_is_smaller_than_multi_turn_paradigm():
    """Sanity check: the single-turn estimate is materially smaller than
    the prior multi-turn estimate that included a 5-turn conditioning
    accumulator."""
    args = _exp3a_args()
    calls = _estimate_call_token_sequence(args, {"intensity_levels": [1]})
    in_tok, _ = calls[0]
    sys_tok = _TOKENS_PER_TURN["system_prompt"]
    cq = _TOKENS_PER_TURN["conditioning_q"]
    ca = _TOKENS_PER_TURN["conditioning_a"]
    cfb = _TOKENS_PER_TURN["conditioning_feedback"]
    multi_turn_post_conditioning = sys_tok + 5 * (cq + ca + cfb)
    assert in_tok < multi_turn_post_conditioning
