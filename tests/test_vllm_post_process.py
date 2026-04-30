from __future__ import annotations

from dataclasses import dataclass

from autocrat.vllm_post_process import PostProcessConfig, post_process_vllm_output


@dataclass(frozen=True)
class _FakeLogProb:
    logprob: float


class _DummyTokenizer:
    eos_token_id = 999

    def __init__(self) -> None:
        self._pieces = {
            11: "<think>",
            12: "Step 1:",
            13: "</think>",
            14: " FINAL_ANSWER: 4",
            21: "A",
            22: "B",
            23: "C",
            24: "D",
            25: "E",
        }

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self._pieces.get(int(tid), f"<{tid}>") for tid in token_ids)


def test_post_process_vllm_output_emits_new_boundary_fields() -> None:
    tokenizer = _DummyTokenizer()
    token_ids = [11, 12, 13, 14]
    logprobs = [
        {11: _FakeLogProb(-0.05), 21: _FakeLogProb(-1.0), 22: _FakeLogProb(-2.0), 23: _FakeLogProb(-3.0), 24: _FakeLogProb(-4.0)},
        {21: _FakeLogProb(-0.10), 12: _FakeLogProb(-0.20), 22: _FakeLogProb(-1.5), 23: _FakeLogProb(-2.0), 24: _FakeLogProb(-3.0)},
        {13: _FakeLogProb(-0.30), 21: _FakeLogProb(-0.35), 22: _FakeLogProb(-1.2), 23: _FakeLogProb(-1.8), 24: _FakeLogProb(-2.2)},
        {14: _FakeLogProb(-0.05), 21: _FakeLogProb(-1.0), 22: _FakeLogProb(-2.0), 23: _FakeLogProb(-3.0), 24: _FakeLogProb(-4.0)},
    ]

    result = post_process_vllm_output(
        prediction="<think>Step 1:</think>\nFINAL_ANSWER: 4",
        token_ids=token_ids,
        logprobs_list=logprobs,
        tokenizer=tokenizer,
        config=PostProcessConfig(boundary_min_tokens=1, max_tokens=8, info_mode=3, cot_mode=2, category="math"),
    )

    assert len(result["boundary_states"]) == 2
    first = result["boundary_states"][0]
    second = result["boundary_states"][1]

    assert isinstance(first["topk_mass"], float)
    assert first["topk_mass"] == first["topk_mass_5"]
    assert first["chosen_token_rank"] == 2
    assert first["tokens_since_last_boundary"] == 2
    assert first["segment_index"] == 0
    assert first["info_mode_switch_flag"] is False
    assert first["info_mode_duration"] == 1
    assert first["top1_token_str"]

    assert second["is_answer_zone"] is True
    assert second["distance_to_answer_zone"] >= 0
    assert second["segment_index"] == 1
    assert second["cot_mode_duration"] == 2

    summary = result["summary"]
    assert summary["prediction_char_length"] > 0
    assert summary["prediction_word_length"] > 0
    assert summary["think_zone_token_count"] >= 1
    assert summary["answer_zone_token_count"] >= 1
