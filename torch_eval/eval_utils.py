#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {exc}")

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = PACKAGE_ROOT.parent
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

if __package__ in (None, "", "torch_eval"):
    from autocrat_controller.action_space import JointActionSpace
    from autocrat import math_grading
    from autocrat.trace_collection import infer_correctness
    from autocrat_controller.features import BoundaryFeatureSpec, HashTextVectorizer
    from autocrat_controller.models import MLP
    from autocrat_controller.neighborhood import BoundaryNeighborhoodConfig, BoundaryNeighborhoodSpec, NeighborhoodFeature
    from autocrat_controller.policy import OnlineDecisionPolicy
    from autocrat_controller.slot_memory import SlotMemory
    from scripts.eval_code_exec import (
        HumanEvalItem,
        MBPPItem,
        _eval_humaneval as _exec_eval_humaneval,
        _eval_mbpp as _exec_eval_mbpp,
        _extract_code as _exec_extract_code,
    )
else:
    from ..autocrat_controller.action_space import JointActionSpace
    from autocrat import math_grading
    from autocrat.trace_collection import infer_correctness
    from ..autocrat_controller.features import BoundaryFeatureSpec, HashTextVectorizer
    from ..autocrat_controller.models import MLP
    from ..autocrat_controller.neighborhood import BoundaryNeighborhoodConfig, BoundaryNeighborhoodSpec, NeighborhoodFeature
    from ..autocrat_controller.policy import OnlineDecisionPolicy
    from ..autocrat_controller.slot_memory import SlotMemory
    from ..scripts.eval_code_exec import (
        HumanEvalItem,
        MBPPItem,
        _eval_humaneval as _exec_eval_humaneval,
        _eval_mbpp as _exec_eval_mbpp,
        _extract_code as _exec_extract_code,
    )


FINAL_ANSWER_MARKER = "FINAL_ANSWER:"
_RE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_RE_FINAL_ANSWER = re.compile(r"FINAL_ANSWER:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
_RE_LAST_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_RE_CHOICE = re.compile(r"\b([A-J])\b")
_RE_CODE_START = re.compile(r"(?m)^\s*(?:FINAL_ANSWER:\s*)?(def |class |from |import )")
_RE_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_RE_THINK_BLOCKS = (
    re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|startofthink\|>.*?<\|endofthink\|>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", flags=re.IGNORECASE | re.DOTALL),
)
_RE_CODE_LINE_START = re.compile(
    r"^\s*(def |class |from |import |@|if __name__ ==|for |while |try:|with |return\b|assert\b|[A-Za-z_]\w*\s*=)"
)
_RE_CODE_ANCHOR = re.compile(
    r"(?m)^\s*(?:FINAL_ANSWER:\s*)?(def |class |from |import |@|if __name__ ==|for |while |try:|with )"
)
_RE_NARRATIVE = re.compile(
    r"^\s*(here|let'?s|i need|the task|solution|explanation|output|final answer|problem|now|first|then)\b",
    flags=re.IGNORECASE,
)
_RE_INSTR_LINE = re.compile(
    r"^\s*(?:[-*]\s+|output rules:|do not\b|if this is\b|otherwise\b|now,\s*write\b|use standard\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EvalItem:
    dataset: str
    problem_id: str
    prompt: str
    reference: str
    category: str
    split: str
    source_path: str
    source_row_index: int
    exec_tests: Tuple[str, ...] = ()
    test_setup_code: str = ""
    entry_point: str = ""
    test_code: str = ""


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    token_count: int
    finish_reason: str
    latency_sec: float
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ProblemResult:
    baseline: str
    dataset: str
    problem_id: str
    correctness: float
    token_count: int
    latency_sec: float
    switches: int
    start_action: Optional[Tuple[int, int]]
    final_action: Optional[Tuple[int, int]]
    prediction: str
    score_proxy: float
    meta: Dict[str, Any]


@dataclass(frozen=True)
class DatasetTokenStats:
    mean: float
    std: float


def _log(message: str) -> None:
    print(message, flush=True)


def _ensure_text(value: Any) -> str:
    return "" if value is None else str(value)


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


def _render_choices_prompt(question: str, options: Sequence[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question, ""]
    for idx, option in enumerate(options):
        lines.append(f"{labels[idx]}. {option}")
    lines.append("")
    lines.append("Answer with one choice letter.")
    return "\n".join(lines)


def _stable_rotate(options: Sequence[str], seed_text: str) -> Tuple[List[str], int]:
    if not options:
        return [], 0
    digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % len(options)
    rotated = list(options[offset:]) + list(options[:offset])
    return rotated, offset


def _make_problem_id(dataset: str, row: Dict[str, Any], index: int) -> str:
    for key in ("id", "problem_id", "task_id", "question_id", "unique_id", "ID", "Record ID"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{dataset}_{index}"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError(f"Unsupported dataset payload at {path}: expected list/dict JSON.")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object at {path} row {idx}")
        out.append(item)
    return out


def _repeat_ngram_ratio(token_ids: Sequence[int], *, n: int = 3, window: int = 128) -> float:
    ids = list(int(x) for x in token_ids[-max(1, int(window)):])
    if len(ids) < n:
        return 0.0
    seen: set[Tuple[int, ...]] = set()
    repeats = 0
    total = 0
    for i in range(len(ids) - n + 1):
        total += 1
        ng = tuple(ids[i : i + n])
        if ng in seen:
            repeats += 1
        seen.add(ng)
    return float(repeats) / float(max(1, total))


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def _extract_per_token_logprob_features(
    logprobs_list: Sequence[Any],
    *,
    eos_token_id: Optional[int],
    token_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, float]]:
    features: List[Dict[str, float]] = []
    if not logprobs_list:
        return features
    for step_logprobs in logprobs_list:
        if not step_logprobs:
            features.append(
                {
                    "entropy": 0.0,
                    "margin": 0.0,
                    "top1_prob": 0.0,
                    "top2_prob": 0.0,
                    "topk_mass": 0.0,
                    "eos_prob": 0.0,
                    "eos_rank": 51.0,
                }
            )
            continue
        items = [(int(tid), _logprob_value(lp)) for tid, lp in step_logprobs.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        probs = [math.exp(lp) for _, lp in items]
        top1_prob = probs[0] if len(probs) >= 1 else 0.0
        top2_prob = probs[1] if len(probs) >= 2 else 0.0
        margin = (items[0][1] - items[1][1]) if len(items) >= 2 else 0.0
        topk_mass = float(sum(probs[:5]))
        entropy = 0.0
        for p in probs:
            if p > 1e-12:
                entropy -= p * math.log(p)
        eos_prob = 0.0
        eos_rank = float(len(items) + 1)
        if eos_token_id is not None:
            for rank, (tid, lp) in enumerate(items):
                if tid == int(eos_token_id):
                    eos_prob = math.exp(lp)
                    eos_rank = float(rank + 1)
                    break
        features.append(
            {
                "entropy": float(entropy),
                "margin": float(margin),
                "top1_prob": float(top1_prob),
                "top2_prob": float(top2_prob),
                "topk_mass": float(topk_mass),
                "eos_prob": float(eos_prob),
                "eos_rank": float(eos_rank),
            }
        )
    return features


def _infer_lambda(scored_rows: Sequence[Mapping[str, Any]]) -> float:
    estimates: List[float] = []
    for row in scored_rows:
        token_z = float(row.get("token_z", 0.0) or 0.0)
        if abs(token_z) < 1e-8:
            continue
        correctness = float(row.get("correctness", row.get("is_correct", 0.0)) or 0.0)
        score = float(row.get("score", correctness) or correctness)
        estimates.append((correctness - score) / token_z)
    if not estimates:
        return 0.0
    return float(np.median(np.asarray(estimates, dtype=np.float32)))


def _compute_dataset_token_stats(scored_rows: Sequence[Mapping[str, Any]]) -> Dict[str, DatasetTokenStats]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for row in scored_rows:
        buckets[str(row["dataset"])].append(int(row.get("token_count", 0) or 0))
    stats: Dict[str, DatasetTokenStats] = {}
    all_values: List[int] = []
    for dataset, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        mean = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size else 1.0
        stats[dataset] = DatasetTokenStats(mean=mean, std=(std if std > 1e-6 else 1.0))
        all_values.extend(values)
    all_arr = np.asarray(all_values, dtype=np.float32)
    all_mean = float(all_arr.mean()) if all_arr.size else 0.0
    all_std = float(all_arr.std()) if all_arr.size else 1.0
    stats["__global__"] = DatasetTokenStats(mean=all_mean, std=(all_std if all_std > 1e-6 else 1.0))
    return stats


def _score_proxy(
    *,
    dataset: str,
    correctness: float,
    token_count: int,
    lambda_penalty: float,
    dataset_token_stats: Mapping[str, DatasetTokenStats],
) -> float:
    stats = dataset_token_stats.get(dataset, dataset_token_stats["__global__"])
    token_z = (float(token_count) - float(stats.mean)) / float(stats.std)
    return float(correctness) - (float(lambda_penalty) * token_z)


def _resolve_path(cfg_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (cfg_path.parent / candidate).resolve()


def _resolve_dataset_source(cfg_path: Path, dataset_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    rooted = (dataset_root / candidate).resolve()
    if rooted.exists():
        return rooted
    return _resolve_path(cfg_path, value)


def _load_items_for_dataset(dataset: str, loader: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    for idx, row in enumerate(rows, start=1):
        if loader == "gsm8k":
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_ensure_text(row.get("question", "")),
                    reference=_ensure_text(row.get("answer", "")),
                    category="math",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "svamp":
            body = _ensure_text(row.get("Body", "")).strip()
            question = _ensure_text(row.get("Question", "")).strip()
            prompt = body if not question else f"{body}\n{question}" if body else question
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=prompt,
                    reference=_ensure_text(row.get("Answer", "")),
                    category="math",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "math_500":
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_ensure_text(row.get("unique_id", "")) or _make_problem_id(dataset, row, idx),
                    prompt=_ensure_text(row.get("problem", "")),
                    reference=_ensure_text(row.get("answer", row.get("solution", ""))),
                    category="math",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "arc_challenge":
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_render_arc_prompt(_ensure_text(row.get("question", "")), row.get("choices")),
                    reference=_ensure_text(row.get("answerKey", "")),
                    category="logic",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "mmlu_pro":
            options = [str(x) for x in row.get("options", [])] if isinstance(row.get("options", []), list) else []
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_render_choices_prompt(_ensure_text(row.get("question", "")), options),
                    reference=_ensure_text(row.get("answer", "")),
                    category="logic",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "gpqa_diamond":
            correct = _ensure_text(row.get("Correct Answer", ""))
            incorrect = [
                _ensure_text(row.get("Incorrect Answer 1", "")),
                _ensure_text(row.get("Incorrect Answer 2", "")),
                _ensure_text(row.get("Incorrect Answer 3", "")),
            ]
            options, offset = _stable_rotate([correct, *incorrect], _ensure_text(row.get("Record ID", "")) or f"{dataset}_{idx}")
            ref_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[(0 - offset) % 4]
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_render_choices_prompt(_ensure_text(row.get("Question", "")), options),
                    reference=ref_letter,
                    category="logic",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "mbpp":
            tests_raw = row.get("test_list", row.get("tests", []))
            tests: List[str] = []
            if isinstance(tests_raw, list):
                tests = [str(x) for x in tests_raw if str(x).strip()]
            elif isinstance(tests_raw, str) and tests_raw.strip():
                tests = [tests_raw]
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_ensure_text(row.get("text", "")),
                    reference=_ensure_text(row.get("code", "")),
                    category="code",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                    exec_tests=tuple(tests),
                    test_setup_code=_ensure_text(row.get("test_setup_code", "")),
                )
            )
        elif loader == "aime2024":
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_ensure_text(row.get("problem", "")),
                    reference=_ensure_text(row.get("solution", "")),
                    category="math",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "aime2025":
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_ensure_text(row.get("problem", "")),
                    reference=_ensure_text(row.get("answer", "")),
                    category="math",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                )
            )
        elif loader == "humaneval":
            items.append(
                EvalItem(
                    dataset=dataset,
                    problem_id=_make_problem_id(dataset, row, idx),
                    prompt=_ensure_text(row.get("prompt", "")),
                    reference=_ensure_text(row.get("canonical_solution", "")),
                    category="code",
                    split=split,
                    source_path=str(source_path),
                    source_row_index=idx,
                    entry_point=_ensure_text(row.get("entry_point", "")),
                    test_code=_ensure_text(row.get("test", "")),
                )
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported loader: {loader}")
    return items


def _load_items(cfg_path: Path, cfg: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, List[EvalItem]]:
    dataset_root = Path(str(cfg.get("dataset_root", ""))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = _resolve_path(cfg_path, str(dataset_root))
    result: Dict[str, List[EvalItem]] = {}
    allow = set(args.datasets or [])
    hard_limit = args.max_items_per_dataset
    for spec in cfg.get("datasets", []):
        if spec.get("enabled", True) is False:
            continue
        dataset = str(spec["dataset"])
        if allow and dataset not in allow:
            continue
        source_path = _resolve_dataset_source(cfg_path, dataset_root, str(spec["source"]))
        rows = _load_rows(source_path)
        offset = int(spec.get("per_dataset_offset", 0) or 0)
        if offset > 0:
            rows = rows[offset:]
        limit = spec.get("per_dataset_limit")
        if hard_limit is not None:
            limit = hard_limit if limit is None else min(int(limit), int(hard_limit))
        if limit is not None:
            rows = rows[: int(limit)]
        result[dataset] = _load_items_for_dataset(
            dataset,
            str(spec["loader"]),
            rows,
            split=str(spec.get("split", "test")),
            source_path=source_path,
        )
    return result


def _render_prompt(prompt_template: str, item: EvalItem) -> str:
    return prompt_template.format(prompt=item.prompt)


def _load_action_space(payload: Mapping[str, Any]) -> JointActionSpace:
    cfg = payload["action_space"]
    return JointActionSpace(
        info_values=tuple(int(x) for x in cfg["info_values"]),
        cot_values=tuple(int(x) for x in cfg["cot_values"]),
        cot_token_budgets={int(k): int(v) for k, v in cfg["cot_token_budgets"].items()},
    )


def _load_slot_memory(payload: Mapping[str, Any]) -> SlotMemory:
    slot = payload["slot_memory"]
    return SlotMemory(
        centers=np.asarray(slot["centers"], dtype=np.float32),
        priors=np.asarray(slot["priors"], dtype=np.float32),
        temperature=float(slot["temperature"]),
    )


def _load_boundary_spec(payload: Mapping[str, Any]) -> BoundaryFeatureSpec:
    spec = payload["boundary_spec"]
    return BoundaryFeatureSpec(
        boundary_kinds=tuple(spec["boundary_kinds"]),
        prompt_log_mean=float(spec["prompt_log_mean"]),
        prompt_log_std=float(spec["prompt_log_std"]),
        generated_log_mean=float(spec["generated_log_mean"]),
        generated_log_std=float(spec["generated_log_std"]),
        entropy_mean=float(spec["entropy_mean"]),
        entropy_std=float(spec["entropy_std"]),
        margin_mean=float(spec["margin_mean"]),
        margin_std=float(spec["margin_std"]),
        top1_prob_mean=float(spec["top1_prob_mean"]),
        top1_prob_std=float(spec["top1_prob_std"]),
        top2_prob_mean=float(spec["top2_prob_mean"]),
        top2_prob_std=float(spec["top2_prob_std"]),
        topk_mass_mean=float(spec["topk_mass_mean"]),
        topk_mass_std=float(spec["topk_mass_std"]),
        eos_prob_mean=float(spec["eos_prob_mean"]),
        eos_prob_std=float(spec["eos_prob_std"]),
        eos_rank_mean=float(spec["eos_rank_mean"]),
        eos_rank_std=float(spec["eos_rank_std"]),
        repeat_ngram_mean=float(spec["repeat_ngram_mean"]),
        repeat_ngram_std=float(spec["repeat_ngram_std"]),
        progress_mean=float(spec["progress_mean"]),
        progress_std=float(spec["progress_std"]),
        remaining_budget_mean=float(spec["remaining_budget_mean"]),
        remaining_budget_std=float(spec["remaining_budget_std"]),
        segment_progress_mean=float(spec["segment_progress_mean"]),
        segment_progress_std=float(spec["segment_progress_std"]),
    )


def _load_model(bundle: Mapping[str, Any], key: str) -> MLP:
    model_cfg = bundle[key]
    model = MLP(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        dropout=0.0,
    )
    model.load_state_dict(model_cfg["state_dict"])
    model.eval()
    return model


def _load_neighborhood_spec(payload: Mapping[str, Any]) -> BoundaryNeighborhoodSpec:
    raw = payload["boundary_neighborhood"]
    cfg = raw["config"]
    config = BoundaryNeighborhoodConfig(
        features=tuple(
            NeighborhoodFeature(
                name=str(item["name"]),
                kind=str(item["kind"]),
                weight=float(item["weight"]),
                enabled=bool(item.get("enabled", True)),
            )
            for item in cfg["features"]
        ),
        top_k=int(cfg["top_k"]),
        kernel_temperature=float(cfg["kernel_temperature"]),
        coarse_bucket_strategy=str(cfg.get("coarse_bucket_strategy", "none")),
        progress_bucket_count=int(cfg.get("progress_bucket_count", 16)),
        progress_bucket_radius=int(cfg.get("progress_bucket_radius", 1)),
        boundary_index_bucket_size=int(cfg.get("boundary_index_bucket_size", 8)),
        boundary_index_bucket_radius=int(cfg.get("boundary_index_bucket_radius", 1)),
        require_same_answer_zone=bool(cfg.get("require_same_answer_zone", True)),
        require_same_boundary_kind=bool(cfg.get("require_same_boundary_kind", False)),
        min_kernel_weight=float(cfg.get("min_kernel_weight", 1e-4)),
    )
    numeric_stats = {
        str(name): (float(stats["mean"]), float(stats["std"]))
        for name, stats in raw["numeric_stats"].items()
    }
    return BoundaryNeighborhoodSpec(config=config, numeric_stats=numeric_stats)


def _torch_load_artifact(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _best_static_action(supervision_dir: Path) -> Tuple[int, int]:
    scored = _load_jsonl(supervision_dir / "offline_supervision" / "scored_traces.jsonl")
    buckets: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for row in scored:
        buckets[(int(row["info_mode"]), int(row["cot_mode"]))].append(float(row.get("score", row.get("correctness", 0.0))))
    ranked = sorted(((action, float(np.mean(scores))) for action, scores in buckets.items()), key=lambda x: x[1], reverse=True)
    return ranked[0][0]


def _category_key(category: Any) -> str:
    text = str(category or "").strip().lower()
    return text or "unknown"


def _dataset_category_key(dataset: Any) -> str:
    name = str(dataset or "").strip().lower()
    if name in {"mbpp", "humaneval"}:
        return "code"
    if name in {"arc_challenge", "gpqa_diamond", "mmlu_pro"}:
        return "logic"
    if name in {"gsm8k", "svamp", "math_500", "aime2024", "aime2025"}:
        return "math"
    return "unknown"


def _scored_row_category_key(row: Mapping[str, Any]) -> str:
    category = _category_key(row.get("category", ""))
    if category != "unknown":
        return category
    return _dataset_category_key(row.get("dataset", ""))


def _best_static_actions_by_category(supervision_dir: Path) -> Dict[str, Tuple[int, int]]:
    scored = _load_jsonl(supervision_dir / "offline_supervision" / "scored_traces.jsonl")
    buckets: Dict[str, Dict[Tuple[int, int], List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in scored:
        category = _scored_row_category_key(row)
        action = (int(row["info_mode"]), int(row["cot_mode"]))
        score = float(row.get("score", row.get("correctness", 0.0)))
        buckets[category][action].append(score)

    best_by_category: Dict[str, Tuple[int, int]] = {}
    for category, action_scores in buckets.items():
        ranked = sorted(
            ((action, float(np.mean(scores))) for action, scores in action_scores.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        if ranked:
            best_by_category[category] = ranked[0][0]
    return best_by_category


def _extract_final_answer_tail(text: str) -> str:
    m = _RE_FINAL_ANSWER.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def _extract_choice_answer(text: str) -> str:
    final_tail = _extract_final_answer_tail(text).strip().upper()
    if final_tail:
        m = re.match(r"^[\(\[]?([A-J])[\)\]\.\,\:\-]?$", final_tail)
        if m:
            return m.group(1).upper()
    m = _RE_CHOICE.search(final_tail)
    if m:
        return m.group(1)
    return ""


def _extract_last_boxed_content(text: str) -> Optional[str]:
    matches = _RE_BOXED.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_numeric_text(text: str) -> Optional[str]:
    matches = _RE_LAST_NUMBER.findall(text.replace(",", ""))
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    try:
        number = float(value)
    except Exception:
        return None
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return str(number)


def _normalize_math_text(text: str) -> str:
    s = _extract_final_answer_tail(text).strip()
    if not s:
        return ""
    if "####" in s:
        tail = s.split("####")[-1].strip()
        number = _normalize_numeric_text(tail)
        if number is not None:
            return number
        return re.sub(r"\s+", "", tail).strip("$")
    boxed = _extract_last_boxed_content(s)
    if boxed:
        return re.sub(r"\s+", "", boxed).strip("$")
    cleaned = re.sub(r"\s+", "", s).strip("$")
    if cleaned and len(cleaned) <= 80 and (("\\" in cleaned) or re.search(r"[A-Za-z]", cleaned)):
        return cleaned
    number = _normalize_numeric_text(s)
    if number is not None:
        return number
    return cleaned


def _extract_math_answer_candidate(text: str) -> str:
    s = _extract_final_answer_tail(text).strip()
    if not s:
        return ""
    boxed = _extract_last_boxed_content(s)
    if boxed:
        return boxed.strip()
    return s


def _gsm8k_correct(prediction: str, reference: str) -> float:
    pred = _normalize_math_text(prediction)
    gold = _normalize_math_text(reference)
    return 1.0 if pred and gold and pred == gold else 0.0


def _math_correct(prediction: str, reference: str) -> float:
    pred = _normalize_math_text(prediction)
    gold = _normalize_math_text(reference)
    return 1.0 if pred and gold and pred == gold else 0.0


def _math500_correct(prediction: str, reference: str) -> float:
    if math_grading.is_available():
        pred_candidate = _extract_math_answer_candidate(prediction)
        gold_candidate = _extract_math_answer_candidate(reference)
        if pred_candidate and gold_candidate:
            return 1.0 if math_grading.grade_answer(pred_candidate, gold_candidate) else 0.0
    return _math_correct(prediction, reference)


def _arc_correct(prediction: str, reference: str) -> float:
    pred = _extract_choice_answer(prediction)
    return 1.0 if pred == str(reference).strip().upper() else 0.0


def _choice_correct(prediction: str, reference: str) -> float:
    pred = _extract_choice_answer(prediction)
    return 1.0 if pred == str(reference).strip().upper() else 0.0


def _extract_code(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = _strip_reasoning_noise(_extract_final_answer_tail(s))
    s = _strip_leading_instruction_lines(s)

    candidates: List[str] = [s]
    close_idx = s.lower().rfind("</think>")
    if close_idx >= 0:
        after_close = s[close_idx + len("</think>") :].strip()
        if after_close:
            candidates.append(after_close)
    candidates.extend(_RE_FENCE.findall(s))

    first_code = _candidate_from_first_code_line(s)
    if first_code:
        candidates.append(first_code)
    first_anchor = _candidate_from_first_code_anchor(s)
    if first_anchor:
        candidates.append(first_anchor)
    code_like = _candidate_code_like_only(s)
    if code_like:
        candidates.append(code_like)

    best = ""
    best_score = (-1, -1, -1, -1)
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_code_candidate(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        score = _score_code_candidate(normalized)
        if score > best_score:
            best_score = score
            best = normalized
    return best


def _strip_reasoning_noise(text: str) -> str:
    out = text or ""
    for pattern in _RE_THINK_BLOCKS:
        out = pattern.sub(" ", out)
    close_idx = out.lower().rfind("</think>")
    if close_idx >= 0:
        tail = out[close_idx + len("</think>") :].strip()
        if tail:
            return tail
    return out


def _normalize_code_candidate(code: str) -> str:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in {"```", "python", "py"}:
            continue
        if stripped.startswith("FINAL_ANSWER:"):
            line = line[line.find(":") + 1 :].lstrip()
        cleaned.append(line.rstrip())
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _strip_leading_instruction_lines(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    idx = 0
    while idx < len(lines) and (not lines[idx].strip() or _RE_INSTR_LINE.match(lines[idx])):
        idx += 1
    return "\n".join(lines[idx:]).strip()


def _looks_code_line(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped.strip():
        return True
    if _RE_CODE_LINE_START.match(stripped):
        return True
    if stripped.lstrip().startswith("#"):
        return True
    if stripped.startswith((" ", "\t")):
        return True
    if stripped.strip().endswith((",", ":", ")", "]", "}", "\\")):
        return True
    return any(token in stripped for token in ["==", "!=", "<=", ">=", " and ", " or ", " in ", " not in ", " is "])


def _candidate_from_first_code_line(text: str) -> str:
    lines = text.splitlines()
    start: Optional[int] = None
    for idx, line in enumerate(lines):
        if _RE_CODE_LINE_START.match(line):
            start = idx
            break
    if start is None:
        return ""
    candidate = lines[start:]
    while candidate and _RE_NARRATIVE.match(candidate[-1]) and not _looks_code_line(candidate[-1]):
        candidate.pop()
    return "\n".join(candidate).strip()


def _candidate_from_first_code_anchor(text: str) -> str:
    match = _RE_CODE_ANCHOR.search(text)
    if not match:
        return ""
    return text[match.start() :].strip()


def _candidate_code_like_only(text: str) -> str:
    kept: List[str] = []
    started = False
    for line in text.splitlines():
        if _looks_code_line(line):
            kept.append(line)
            started = True
            continue
        if started and _RE_NARRATIVE.match(line):
            break
        if started:
            kept.append(line)
    return "\n".join(kept).strip()


def _score_code_candidate(code: str) -> Tuple[int, int, int, int]:
    normalized = _normalize_code_candidate(code)
    if not normalized:
        return (0, 0, 0, 0)
    parse_ok = 0
    try:
        ast.parse(normalized)
        parse_ok = 1
    except Exception:
        parse_ok = 0
    has_core = 1 if re.search(r"^\s*(def |class |from |import )", normalized, flags=re.MULTILINE) else 0
    non_empty = sum(1 for line in normalized.splitlines() if line.strip())
    return (parse_ok, has_core, non_empty, len(normalized))


def _top_level_callable_names(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def _expected_callable_names_from_tests(tests: Sequence[str]) -> List[str]:
    names: List[str] = []
    pattern = re.compile(
        r"""assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|==|!=|<=|>=|<|>|is\b|in\b)""",
        flags=re.IGNORECASE,
    )
    for test in tests:
        for match in pattern.finditer(str(test)):
            name = match.group(1)
            if name and name not in names:
                names.append(name)
    return names


def _append_missing_aliases(code: str, expected_names: Sequence[str]) -> str:
    if not expected_names:
        return code
    top_level = _top_level_callable_names(code)
    if len(top_level) != 1:
        return code
    src = top_level[0]
    present = set(top_level)
    aliases = [f"{name} = {src}" for name in expected_names if name not in present and name != src]
    if not aliases:
        return code
    return code.rstrip() + "\n\n" + "\n".join(aliases) + "\n"


def _evaluate_prediction(item: EvalItem, prediction: str, *, timeout_sec: float, python_bin: str) -> Tuple[float, str]:
    if item.dataset == "mbpp":
        mbpp_item = MBPPItem(
            problem_id=str(item.problem_id),
            prompt=str(item.prompt),
            tests=[str(x) for x in item.exec_tests],
            test_setup_code=str(item.test_setup_code),
        )
        code = _exec_extract_code(str(prediction))
        ok, err = _exec_eval_mbpp(code, mbpp_item, timeout_sec=timeout_sec, python_bin=python_bin)
        return (1.0 if ok else 0.0), err
    if item.dataset == "humaneval":
        humaneval_item = HumanEvalItem(
            task_id=str(item.problem_id),
            prompt=str(item.prompt),
            test=str(item.test_code),
            entry_point=str(item.entry_point),
        )
        code = _exec_extract_code(str(prediction))
        ok, err = _exec_eval_humaneval(code, humaneval_item, timeout_sec=timeout_sec, python_bin=python_bin)
        return (1.0 if ok else 0.0), err
    row = {
        "dataset": str(item.dataset),
        "problem_id": str(item.problem_id),
        "category": str(item.category),
        "prediction": str(prediction),
        "reference": str(item.reference),
    }
    try:
        return float(infer_correctness(row, code_policy="execute_tests")), ""
    except Exception as exc:
        return 0.0, f"eval_error: {exc}"


def _cot_budget(cot_budgets: Mapping[str, Any], cot_mode: int) -> int:
    return int(cot_budgets[str(cot_mode)] if str(cot_mode) in cot_budgets else cot_budgets[cot_mode])


def _static_max_tokens_for_action(
    *,
    cot_budgets: Mapping[str, Any],
    cot_mode: int,
    runtime_cfg: Mapping[str, Any],
) -> int:
    answer_budget = int(runtime_cfg.get("static_answer_completion_budget", 16384) or 0)
    return max(1, _cot_budget(cot_budgets, int(cot_mode)) + max(0, answer_budget))


def _build_boundary_row(
    *,
    item: EvalItem,
    generated_tokens: int,
    think_budget: int,
    think_used: int,
    in_answer_zone: bool,
    boundary_kind: str,
    entropy: float,
    margin: float,
    top1_prob: float,
    top2_prob: float,
    topk_mass: float,
    eos_prob: float,
    eos_rank: float,
    repeat_ngram_ratio: float,
) -> Dict[str, Any]:
    total_budget = max(1, think_budget + generated_tokens + 1)
    progress_ratio = min(1.0, float(generated_tokens) / float(total_budget))
    remaining_ratio = 0.0 if in_answer_zone or think_budget <= 0 else max(0.0, float(think_budget - think_used) / float(max(1, think_budget)))
    return {
        "prompt": item.prompt,
        "dataset": item.dataset,
        "problem_id": item.problem_id,
        "boundary_index": 0,
        "state_features": {
            "generated_tokens": int(generated_tokens),
            "progress_ratio": float(progress_ratio),
            "remaining_budget_ratio": float(remaining_ratio),
            "segment_progress_ratio": 1.0,
            "entropy": float(entropy),
            "margin": float(margin),
            "top1_prob": float(top1_prob),
            "top2_prob": float(top2_prob),
            "topk_mass": float(topk_mass),
            "eos_prob": float(eos_prob),
            "eos_rank": float(eos_rank),
            "repeat_ngram_ratio": float(repeat_ngram_ratio),
            "boundary_kind": str(boundary_kind),
            "is_answer_zone": bool(in_answer_zone),
            "is_in_think_zone": bool(not in_answer_zone),
            "is_code_mode": bool(item.category == "code"),
        },
    }


def _find_chunk_boundary(text: str, *, min_chars: int) -> int:
    if len(text) <= int(min_chars):
        return len(text)
    candidates = []
    for match in re.finditer(r"[\n.!?;:]", text):
        if match.end() >= int(min_chars):
            candidates.append(match.end())
    return candidates[-1] if candidates else len(text)


def _truncate_chunk_to_boundary(
    tokenizer: Any,
    token_ids: Sequence[int],
    text: str,
    *,
    min_chars: int,
) -> Tuple[str, int]:
    if not token_ids:
        return "", 0
    cutoff_chars = _find_chunk_boundary(text, min_chars=min_chars)
    if cutoff_chars >= len(text):
        return str(text), len(token_ids)
    pieces: List[str] = []
    cumulative = ""
    accepted_tokens = 0
    for tid in token_ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
        candidate = cumulative + str(piece)
        if len(candidate) > cutoff_chars and accepted_tokens > 0:
            break
        cumulative = candidate
        pieces.append(str(piece))
        accepted_tokens += 1
        if len(cumulative) >= cutoff_chars:
            break
    if accepted_tokens <= 0:
        return str(text), len(token_ids)
    return "".join(pieces), accepted_tokens


def _resolve_start_action(
    *,
    strategy: str,
    prompt: str,
    policy: OnlineDecisionPolicy,
    action_space: JointActionSpace,
    best_static_action: Tuple[int, int],
    category: str = "",
    category_best_static_actions: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    if strategy == "global_best_static":
        return (int(best_static_action[0]), int(best_static_action[1]))
    if strategy == "category_best_static":
        category_actions = category_best_static_actions or {}
        action = category_actions.get(_category_key(category), best_static_action)
        return (int(action[0]), int(action[1]))
    if strategy == "prior_model":
        prior_scores = policy.score_initial_actions(prompt)
        action = action_space.index_to_action(int(np.argmax(prior_scores)))
        return (int(action[0]), int(action[1]))
    raise ValueError(f"Unsupported start_action_strategy: {strategy}")
