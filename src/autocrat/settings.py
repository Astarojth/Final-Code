from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class InfoModeSetting:
    level: int
    temperature: float
    top_p: float
    description: str


@dataclass(frozen=True)
class CoTModeSetting:
    level: int
    min_steps: int
    max_steps: int
    max_thinking_tokens: int
    description: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    category: str
    hf_dataset: str
    subset: str | None
    train_split: str
    eval_split: str
    prompt_field: str
    answer_field: str
    notes: str


INFO_MODE_TABLE: Dict[int, InfoModeSetting] = {
    1: InfoModeSetting(1, 0.0, 0.95, "deterministic"),
    2: InfoModeSetting(2, 0.3, 0.95, "low temperature"),
    3: InfoModeSetting(3, 0.7, 0.95, "medium temperature"),
    4: InfoModeSetting(4, 1.0, 0.95, "high temperature"),
}


COT_MODE_TABLE: Dict[int, CoTModeSetting] = {
    0: CoTModeSetting(0, 0, 0, 0, "no thinking"),
    1: CoTModeSetting(1, 0, 0, 64, "short thinking"),
    2: CoTModeSetting(2, 0, 0, 256, "medium thinking"),
    3: CoTModeSetting(3, 0, 0, 1024, "long thinking"),
    4: CoTModeSetting(4, 0, 0, 4096, "very long thinking"),
}


def cot_thinking_token_budget(cot: CoTModeSetting, *, max_new_tokens_per_step: int) -> int:
    budget = int(getattr(cot, "max_thinking_tokens", 0) or 0)
    if budget > 0:
        return budget
    return max(8, int(cot.max_steps) * int(max_new_tokens_per_step))


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        key="gsm8k",
        category="math",
        hf_dataset="openai/gsm8k",
        subset="main",
        train_split="train",
        eval_split="test",
        prompt_field="question",
        answer_field="answer",
        notes="Primary arithmetic reasoning benchmark.",
    ),
    "arc_challenge": DatasetSpec(
        key="arc_challenge",
        category="logic",
        hf_dataset="allenai/ai2_arc",
        subset="ARC-Challenge",
        train_split="train",
        eval_split="test",
        prompt_field="question",
        answer_field="answerKey",
        notes="Classical multi-choice reasoning benchmark.",
    ),
    "mbpp": DatasetSpec(
        key="mbpp",
        category="code",
        hf_dataset="mbpp",
        subset=None,
        train_split="train",
        eval_split="test",
        prompt_field="text",
        answer_field="code",
        notes="Python programming reasoning benchmark.",
    ),
    "math_500": DatasetSpec(
        key="math_500",
        category="math",
        hf_dataset="HuggingFaceH4/MATH-500",
        subset=None,
        train_split="test",
        eval_split="test",
        prompt_field="problem",
        answer_field="answer",
        notes="MATH-500 symbolic math evaluation benchmark.",
    ),
    "gpqa_diamond": DatasetSpec(
        key="gpqa_diamond",
        category="qa",
        hf_dataset="Idavidrein/gpqa",
        subset="gpqa_diamond",
        train_split="train",
        eval_split="train",
        prompt_field="Question",
        answer_field="Correct Answer",
        notes="GPQA-Diamond challenging QA benchmark.",
    ),
    "humaneval": DatasetSpec(
        key="humaneval",
        category="code",
        hf_dataset="openai/openai_humaneval",
        subset=None,
        train_split="test",
        eval_split="test",
        prompt_field="prompt",
        answer_field="canonical_solution",
        notes="Code generation benchmark; evaluation uses pass@k.",
    ),
}


def validate_mode_tables() -> None:
    info_levels = sorted(INFO_MODE_TABLE.keys())
    cot_levels = sorted(COT_MODE_TABLE.keys())
    if info_levels != [1, 2, 3, 4]:
        raise ValueError(f"Unexpected sampling levels: {info_levels}")
    if cot_levels != [0, 1, 2, 3, 4]:
        raise ValueError(f"Unexpected budget levels: {cot_levels}")
