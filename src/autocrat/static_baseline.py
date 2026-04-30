from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from .offline_supervision import read_jsonl
from .settings import DATASET_REGISTRY


@dataclass(frozen=True)
class EvalItem:
    dataset: str
    problem_id: str
    prompt: str
    reference: str
    category: str


@dataclass(frozen=True)
class StaticMode:
    info_mode: int
    cot_mode: int
    tag: str


def _ensure_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _render_arc_prompt(question: str, choices: Any) -> str:
    if not isinstance(choices, dict):
        return question
    labels = choices.get("label")
    texts = choices.get("text")
    if not isinstance(labels, list) or not isinstance(texts, list):
        return question
    lines = [question, ""]
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")
    lines.append("")
    lines.append("Answer with one choice letter.")
    return "\n".join(lines)


def _to_eval_item(dataset_key: str, row: Dict[str, Any], index: int) -> EvalItem:
    spec = DATASET_REGISTRY[dataset_key]
    prompt_raw = row.get(spec.prompt_field, "")
    prompt = _ensure_text(prompt_raw)
    if dataset_key in {"arc_challenge", "arc_easy", "commonsenseqa"}:
        prompt = _render_arc_prompt(prompt, row.get("choices"))
    reference = _ensure_text(row.get(spec.answer_field, ""))
    problem_id = str(row.get("id") or row.get("problem_id") or f"{dataset_key}_{index}")
    return EvalItem(
        dataset=dataset_key,
        problem_id=problem_id,
        prompt=prompt,
        reference=reference,
        category=spec.category,
    )


def load_eval_items_from_hf(
    dataset_keys: Iterable[str],
    *,
    per_dataset_limit: int,
    split_kind: str = "eval",
) -> List[EvalItem]:
    all_items: List[EvalItem] = []
    for dataset_key in dataset_keys:
        if dataset_key not in DATASET_REGISTRY:
            raise KeyError(f"Unknown dataset key: {dataset_key}")
        spec = DATASET_REGISTRY[dataset_key]
        split_name = spec.eval_split if split_kind == "eval" else spec.train_split
        ds = load_dataset(spec.hf_dataset, spec.subset, split=split_name)
        limit = min(per_dataset_limit, len(ds))
        for i in range(limit):
            all_items.append(_to_eval_item(dataset_key, dict(ds[i]), i))
    return all_items


def load_eval_items_from_jsonl(path: str | Path) -> List[EvalItem]:
    rows = read_jsonl(path)
    items: List[EvalItem] = []
    for idx, row in enumerate(rows, start=1):
        dataset = str(row.get("dataset", "")).strip()
        problem_id = str(row.get("problem_id", f"row{idx}")).strip()
        prompt = str(row.get("prompt", "")).strip()
        reference = str(row.get("reference", "")).strip()
        category = str(row.get("category", "")).strip().lower()
        if not dataset or not problem_id or not prompt:
            raise ValueError(f"Invalid eval item at row={idx}: missing dataset/problem_id/prompt")
        if not category and dataset in DATASET_REGISTRY:
            category = DATASET_REGISTRY[dataset].category
        items.append(
            EvalItem(
                dataset=dataset,
                problem_id=problem_id,
                prompt=prompt,
                reference=reference,
                category=category or "unknown",
            )
        )
    return items


def parse_static_modes(mode_rows: List[Dict[str, Any]]) -> List[StaticMode]:
    modes: List[StaticMode] = []
    for idx, row in enumerate(mode_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"static_mode_candidates item #{idx} must be an object.")
        info_mode = int(row["info_mode"])
        cot_mode = int(row["cot_mode"])
        tag = str(row.get("tag", f"m{info_mode}_{cot_mode}"))
        modes.append(StaticMode(info_mode=info_mode, cot_mode=cot_mode, tag=tag))
    if not modes:
        raise ValueError("At least one static mode is required.")
    return modes
