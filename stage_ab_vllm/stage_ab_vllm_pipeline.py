#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocrat.config import load_yaml, resolve_path
from autocrat.custom_backend import build_custom_backend
from autocrat.gpu_guard import is_gpu_busy, query_gpu_status
from autocrat.offline_supervision import (
    build_boundary_preference_pairs,
    build_boundary_samples,
    build_preference_pairs,
    find_hard_examples,
    write_jsonl,
)
from autocrat.static_baseline import parse_static_modes
from autocrat.trace_collection import collect_static_traces, infer_correctness
from autocrat.vllm_post_process import PostProcessConfig, post_process_vllm_output
from scripts.eval_code_exec import HumanEvalItem, MBPPItem, _eval_humaneval, _eval_mbpp, _extract_code


@dataclass(frozen=True)
class StageItem:
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


def _set_generation_seed(seed: int, *, touch_torch: bool = True) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    if touch_torch:
        try:
            import torch  # type: ignore

            torch.manual_seed(seed)
            if bool(torch.cuda.is_available()):
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        try:
            from transformers import set_seed  # type: ignore

            set_seed(seed)
        except Exception:
            pass


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(item)
    return rows


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield item


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except FileNotFoundError:
        pass


def _format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _ProgressPrinter:
    def __init__(self, *, total: int, done: int = 0) -> None:
        self.total = max(0, int(total))
        self.done = max(0, int(done))
        self.started_at = time.perf_counter()
        self.last_emit = 0.0

    def update(self, n: int = 1, *, extra: str = "") -> None:
        self.done += int(n)
        now = time.perf_counter()
        if (now - self.last_emit) < 2.0 and self.done < self.total:
            return
        self.last_emit = now
        elapsed = max(1e-9, now - self.started_at)
        rate = self.done / elapsed
        remaining = max(0, self.total - self.done)
        eta = (remaining / rate) if rate > 1e-9 else float("inf")
        pct = (100.0 * self.done / self.total) if self.total > 0 else 100.0
        msg = (
            f"[progress] {self.done}/{self.total} ({pct:.1f}%) "
            f"elapsed={_format_seconds(elapsed)} "
            f"eta={_format_seconds(eta) if eta != float('inf') else '--:--'} "
            f"rate={rate:.2f}/s"
        )
        if extra:
            msg += f" {extra}"
        print(msg, flush=True)


def _scan_existing_raw_rows(path: Path) -> Tuple[set[Tuple[str, str, str]], Dict[str, int]]:
    completed: set[Tuple[str, str, str]] = set()
    counts = {
        "raw_predictions": 0,
        "scored_raw_rows": 0,
        "prediction_only_rows": 0,
        "eval_error_rows": 0,
    }
    for row in _iter_jsonl(path):
        dataset = str(row.get("dataset", ""))
        problem_id = str(row.get("problem_id", ""))
        mode_tag = str(row.get("mode_tag", ""))
        completed.add((dataset, problem_id, mode_tag))
        counts["raw_predictions"] += 1
        status = str(row.get("eval_status", ""))
        if status == "scored":
            counts["scored_raw_rows"] += 1
        elif status == "prediction_only":
            counts["prediction_only_rows"] += 1
        elif status == "eval_error":
            counts["eval_error_rows"] += 1
    return completed, counts


def _aggregate_raw_from_file(path: Path) -> Dict[str, Any]:
    by_dataset: Dict[str, Dict[str, float]] = {}
    for row in _iter_jsonl(path):
        dataset = str(row["dataset"])
        bucket = by_dataset.setdefault(
            dataset,
            {
                "count": 0.0,
                "correct": 0.0,
                "token_count": 0.0,
                "prediction_only": 0.0,
                "eval_error": 0.0,
            },
        )
        bucket["count"] += 1.0
        bucket["token_count"] += float(row.get("token_count", 0) or 0)
        if row.get("eval_status") == "scored":
            bucket["correct"] += float(row.get("is_correct", 0.0) or 0.0)
        elif row.get("eval_status") == "prediction_only":
            bucket["prediction_only"] += 1.0
        elif row.get("eval_status") == "eval_error":
            bucket["eval_error"] += 1.0

    out: Dict[str, Any] = {}
    for dataset, bucket in by_dataset.items():
        count = max(1.0, float(bucket["count"]))
        out[dataset] = {
            "count": int(bucket["count"]),
            "avg_tokens": bucket["token_count"] / count,
            "avg_correctness_scored_rows": bucket["correct"]
            / max(1.0, count - bucket["prediction_only"] - bucket["eval_error"]),
            "prediction_only_rows": int(bucket["prediction_only"]),
            "eval_error_rows": int(bucket["eval_error"]),
        }
    return out


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


def _resolve_output_root(cfg_path: Path, cfg: Dict[str, Any]) -> Path:
    output_root = cfg.get("output_root")
    if output_root:
        return resolve_path(cfg_path, str(output_root))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_path(cfg_path, f"../runs/stage_ab_vllm/stage_a_{ts}")


def _resolve_dataset_source(cfg_path: Path, dataset_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    rooted = (dataset_root / candidate).resolve()
    if rooted.exists():
        return rooted
    return resolve_path(cfg_path, value)


def _make_problem_id(dataset: str, row: Dict[str, Any], index: int) -> str:
    for key in ("id", "problem_id", "task_id", "question_id", "unique_id", "ID", "Record ID"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{dataset}_{index}"


def _load_gsm8k_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        items.append(
            StageItem(
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
    return items


def _load_svamp_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        body = _ensure_text(row.get("Body", "")).strip()
        question = _ensure_text(row.get("Question", "")).strip()
        prompt = body if not question else f"{body}\n{question}" if body else question
        items.append(
            StageItem(
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
    return items


def _load_math500_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        problem_id = _ensure_text(row.get("unique_id", "")) or _make_problem_id(dataset, row, idx)
        items.append(
            StageItem(
                dataset=dataset,
                problem_id=problem_id,
                prompt=_ensure_text(row.get("problem", "")),
                reference=_ensure_text(row.get("answer", row.get("solution", ""))),
                category="math",
                split=split,
                source_path=str(source_path),
                source_row_index=idx,
            )
        )
    return items


def _load_aime2024_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        items.append(
            StageItem(
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
    return items


def _load_aime2025_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        items.append(
            StageItem(
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
    return items


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


def _load_mmlu_pro_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        options_raw = row.get("options", [])
        options = [str(x) for x in options_raw] if isinstance(options_raw, list) else []
        items.append(
            StageItem(
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
    return items


def _load_gpqa_diamond_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        correct = _ensure_text(row.get("Correct Answer", ""))
        incorrect = [
            _ensure_text(row.get("Incorrect Answer 1", "")),
            _ensure_text(row.get("Incorrect Answer 2", "")),
            _ensure_text(row.get("Incorrect Answer 3", "")),
        ]
        all_options = [correct, *incorrect]
        seed_text = _ensure_text(row.get("Record ID", "")) or f"{dataset}_{idx}"
        rotated, offset = _stable_rotate(all_options, seed_text)
        correct_idx = (0 - offset) % len(all_options)
        ref_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[correct_idx]
        items.append(
            StageItem(
                dataset=dataset,
                problem_id=_make_problem_id(dataset, row, idx),
                prompt=_render_choices_prompt(_ensure_text(row.get("Question", "")), rotated),
                reference=ref_letter,
                category="logic",
                split=split,
                source_path=str(source_path),
                source_row_index=idx,
            )
        )
    return items


def _load_arc_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        question = _ensure_text(row.get("question", ""))
        prompt = _render_arc_prompt(question, row.get("choices"))
        items.append(
            StageItem(
                dataset=dataset,
                problem_id=_make_problem_id(dataset, row, idx),
                prompt=prompt,
                reference=_ensure_text(row.get("answerKey", "")),
                category="logic",
                split=split,
                source_path=str(source_path),
                source_row_index=idx,
            )
        )
    return items


def _load_mbpp_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        tests_raw = row.get("test_list", row.get("tests", []))
        tests: List[str] = []
        if isinstance(tests_raw, list):
            tests = [str(x) for x in tests_raw if str(x).strip()]
        elif isinstance(tests_raw, str) and tests_raw.strip():
            tests = [str(tests_raw)]
        items.append(
            StageItem(
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
    return items


def _load_humaneval_items(dataset: str, rows: Sequence[Dict[str, Any]], *, split: str, source_path: Path) -> List[StageItem]:
    items: List[StageItem] = []
    for idx, row in enumerate(rows, start=1):
        items.append(
            StageItem(
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
    return items


def _evaluate_code_prediction(
    item: StageItem,
    prediction: str,
    *,
    timeout_sec: float,
    python_bin: str,
) -> Tuple[float, str]:
    if item.dataset == "mbpp":
        mbpp_item = MBPPItem(
            problem_id=str(item.problem_id),
            prompt=str(item.prompt),
            tests=[str(x) for x in item.exec_tests],
            test_setup_code=str(item.test_setup_code),
        )
        code = _extract_code(str(prediction))
        ok, err = _eval_mbpp(code, mbpp_item, timeout_sec=timeout_sec, python_bin=python_bin)
        return (1.0 if ok else 0.0), str(err or "")
    if item.dataset == "humaneval":
        humaneval_item = HumanEvalItem(
            task_id=str(item.problem_id),
            prompt=str(item.prompt),
            test=str(item.test_code),
            entry_point=str(item.entry_point),
        )
        code = _extract_code(str(prediction))
        ok, err = _eval_humaneval(code, humaneval_item, timeout_sec=timeout_sec, python_bin=python_bin)
        return (1.0 if ok else 0.0), str(err or "")
    raise ValueError(f"Unsupported code execution dataset={item.dataset!r}")


_DATASET_LOADERS = {
    "gsm8k": _load_gsm8k_items,
    "svamp": _load_svamp_items,
    "math_500": _load_math500_items,
    "arc_challenge": _load_arc_items,
    "mmlu_pro": _load_mmlu_pro_items,
    "gpqa_diamond": _load_gpqa_diamond_items,
    "mbpp": _load_mbpp_items,
    "aime2024": _load_aime2024_items,
    "aime2025": _load_aime2025_items,
    "humaneval": _load_humaneval_items,
}


def _load_stage_items(cfg_path: Path, cfg: Dict[str, Any]) -> List[StageItem]:
    dataset_root = Path(str(cfg.get("dataset_root", ""))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = resolve_path(cfg_path, str(dataset_root))
    dataset_specs = cfg.get("datasets")
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise ValueError("Config must provide a non-empty `datasets` list.")

    items: List[StageItem] = []
    for idx, spec in enumerate(dataset_specs, start=1):
        if not isinstance(spec, dict):
            raise ValueError(f"datasets[{idx}] must be an object.")
        if spec.get("enabled", True) is False:
            continue
        dataset = str(spec.get("dataset", "")).strip()
        loader_name = str(spec.get("loader", dataset)).strip()
        source = str(spec.get("source", "")).strip()
        split = str(spec.get("split", "train")).strip() or "train"
        offset = int(spec.get("per_dataset_offset", 0) or 0)
        limit = int(spec.get("per_dataset_limit", cfg.get("per_dataset_limit", 0)) or 0)
        if not dataset or not source:
            raise ValueError(f"datasets[{idx}] missing dataset/source.")
        if loader_name not in _DATASET_LOADERS:
            raise ValueError(f"Unsupported loader={loader_name!r} for dataset={dataset!r}.")
        source_path = _resolve_dataset_source(cfg_path, dataset_root, source)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset source not found: {source_path}")
        rows = _read_rows(source_path)
        if offset > 0:
            rows = rows[offset:]
        dataset_items = _DATASET_LOADERS[loader_name](dataset, rows, split=split, source_path=source_path)
        if limit > 0:
            dataset_items = dataset_items[:limit]
        items.extend(dataset_items)
    return items


def _build_prompt(item: StageItem, template: str, *, use_task_metadata: bool) -> str:
    dataset = item.dataset if use_task_metadata else ""
    problem_id = item.problem_id if use_task_metadata else ""
    base = template.format(dataset=dataset, problem_id=problem_id, prompt=item.prompt)
    if str(item.category).strip().lower() == "code":
        return (
            f"{base}\n\n"
            "Code-task override:\n"
            "- After </think>, immediately output FINAL_ANSWER: followed by the final code solution.\n"
            "- Put the final code directly after FINAL_ANSWER:.\n"
            "- Do not explain the solution.\n"
            "- Do not include prose before or after the code.\n"
            "- Do not add any extra text after the final code.\n"
        )
    return (
        f"{base}\n\n"
        "Non-code-task override:\n"
        "- After </think>, output exactly one final line in the form: FINAL_ANSWER: <answer>\n"
        "- Do not add explanation, derivation, or any extra lines after FINAL_ANSWER.\n"
        "- Keep the final answer line as short as possible.\n"
    )


def _normalize_prediction_for_category(category: str, prediction: str) -> str:
    text = str(prediction or "")
    if str(category).strip().lower() != "code":
        return text
    marker = "FINAL_ANSWER:"
    upper = text.upper()
    pos = upper.rfind(marker)
    if pos >= 0:
        tail = text[pos + len(marker) :].lstrip()
        if tail:
            return tail
    return text


def _answer_trigger_text_for_category(category: str) -> str:
    return "</think>\nFINAL_ANSWER: "


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg) or ("cublas" in msg and "alloc" in msg)


def _maybe_empty_cuda_cache(backend: Any) -> None:
    torch_mod = getattr(backend, "_torch", None)
    if torch_mod is None:
        return
    try:
        if bool(torch_mod.cuda.is_available()):
            torch_mod.cuda.empty_cache()
    except Exception:
        pass


def _aggregate_raw_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_dataset: Dict[str, Dict[str, float]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        bucket = by_dataset.setdefault(
            dataset,
            {
                "count": 0.0,
                "correct": 0.0,
                "token_count": 0.0,
                "prediction_only": 0.0,
                "eval_error": 0.0,
            },
        )
        bucket["count"] += 1.0
        bucket["token_count"] += float(row.get("token_count", 0) or 0)
        if row.get("eval_status") == "scored":
            bucket["correct"] += float(row.get("is_correct", 0.0) or 0.0)
        elif row.get("eval_status") == "prediction_only":
            bucket["prediction_only"] += 1.0
        elif row.get("eval_status") == "eval_error":
            bucket["eval_error"] += 1.0

    out: Dict[str, Any] = {}
    for dataset, bucket in by_dataset.items():
        count = max(1.0, float(bucket["count"]))
        out[dataset] = {
            "count": int(bucket["count"]),
            "avg_tokens": bucket["token_count"] / count,
            "avg_correctness_scored_rows": bucket["correct"] / max(1.0, count - bucket["prediction_only"] - bucket["eval_error"]),
            "prediction_only_rows": int(bucket["prediction_only"]),
            "eval_error_rows": int(bucket["eval_error"]),
        }
    return out


def _empty_offline_outputs(output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "scored_traces": output_dir / "scored_traces.jsonl",
        "hard_examples": output_dir / "hard_examples.jsonl",
        "preference_pairs": output_dir / "preference_pairs.jsonl",
        "boundary_samples": output_dir / "boundary_samples.jsonl",
        "boundary_preference_pairs": output_dir / "boundary_preference_pairs.jsonl",
    }
    for path in files.values():
        write_jsonl(path, [])
    return {key: str(path.resolve()) for key, path in files.items()}


def _prepare_offline_outputs(
    traces: List[Dict[str, Any]],
    *,
    output_dir: Path,
    lambda_token_penalty: float,
    hidden_proj_dim: int,
    token_z_threshold: float,
    min_score_gap: float,
    max_pairs_per_problem: int,
    pair_strategy: str,
    disagreement_reweight: bool,
) -> Dict[str, Any]:
    from autocrat.offline_supervision import score_traces

    if not traces:
        files = _empty_offline_outputs(output_dir)
        summary = {
            "counts": {
                "traces": 0,
                "scored_traces": 0,
                "hard_examples": 0,
                "preference_pairs": 0,
                "boundary_samples": 0,
                "boundary_preference_pairs": 0,
            },
            "files": files,
        }
        return summary

    scored = score_traces(traces, lambda_token_penalty=lambda_token_penalty)
    hard_examples = find_hard_examples(scored, token_z_threshold=token_z_threshold)
    preference_pairs = build_preference_pairs(
        scored,
        min_score_gap=min_score_gap,
        max_pairs_per_problem=max_pairs_per_problem,
        pair_strategy=pair_strategy,
        disagreement_reweight=disagreement_reweight,
    )
    boundary_samples = build_boundary_samples(scored, hidden_proj_dim=hidden_proj_dim)
    boundary_pairs = build_boundary_preference_pairs(
        boundary_samples,
        min_score_gap=min_score_gap,
        max_pairs_per_problem=max_pairs_per_problem,
        pair_strategy=pair_strategy,
        disagreement_reweight=disagreement_reweight,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = write_jsonl(output_dir / "scored_traces.jsonl", scored)
    hard_path = write_jsonl(output_dir / "hard_examples.jsonl", hard_examples)
    pref_path = write_jsonl(output_dir / "preference_pairs.jsonl", preference_pairs)
    boundary_samples_path = write_jsonl(output_dir / "boundary_samples.jsonl", boundary_samples)
    boundary_pairs_path = write_jsonl(output_dir / "boundary_preference_pairs.jsonl", boundary_pairs)

    summary = {
        "counts": {
            "traces": len(traces),
            "scored_traces": len(scored),
            "hard_examples": len(hard_examples),
            "preference_pairs": len(preference_pairs),
            "boundary_samples": len(boundary_samples),
            "boundary_preference_pairs": len(boundary_pairs),
        },
        "files": {
            "scored_traces": str(scored_path.resolve()),
            "hard_examples": str(hard_path.resolve()),
            "preference_pairs": str(pref_path.resolve()),
            "boundary_samples": str(boundary_samples_path.resolve()),
            "boundary_preference_pairs": str(boundary_pairs_path.resolve()),
        },
        "hard_examples": hard_examples,
    }
    return summary


def _hard_items_from_examples(items: Sequence[StageItem], hard_examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = {(str(row.get("dataset", "")), str(row.get("problem_id", ""))) for row in hard_examples}
    out: List[Dict[str, Any]] = []
    for item in items:
        if (item.dataset, item.problem_id) not in keep:
            continue
        out.append(
            {
                "dataset": item.dataset,
                "problem_id": item.problem_id,
                "prompt": item.prompt,
                "reference": item.reference,
                "category": item.category,
                "split": item.split,
                "source_path": item.source_path,
                "source_row_index": item.source_row_index,
            }
        )
    return out


def _hard_items_from_normalized_rows(
    items: Sequence[Dict[str, Any]],
    hard_examples: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    keep = {(str(row.get("dataset", "")), str(row.get("problem_id", ""))) for row in hard_examples}
    out: List[Dict[str, Any]] = []
    for item in items:
        dataset = str(item.get("dataset", ""))
        problem_id = str(item.get("problem_id", ""))
        if (dataset, problem_id) not in keep:
            continue
        out.append(
            {
                "dataset": dataset,
                "problem_id": problem_id,
                "prompt": item.get("prompt"),
                "reference": item.get("reference"),
                "category": item.get("category"),
                "split": item.get("split"),
                "source_path": item.get("source_path"),
                "source_row_index": item.get("source_row_index"),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A local vLLM pipeline for AUTOCRAT.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "stage_a.qwen3_4b_thinking.local.yaml"),
        help="Pipeline config path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run backend.")
    parser.add_argument("--max-items-per-dataset", type=int, default=0, help="Temporary cap for smoke runs.")
    parser.add_argument("--max-gpu-used-ratio", type=float, default=0.98, help="Stop if GPU memory ratio is too high.")
    parser.add_argument("--max-gpu-util", type=int, default=95, help="Stop if GPU utilization is too high.")
    parser.add_argument(
        "--phase",
        choices=("all", "infer", "finalize"),
        default="all",
        help="Run inference only, finalize only, or both.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh run instead of resuming from existing raw_predictions.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    model_cfg = load_yaml(resolve_path(cfg_path, str(cfg.get("model_config"))))

    generation_seed = int(cfg.get("generation_seed", 42) or 42)
    model_cfg.setdefault("sampling_seed", generation_seed)
    runtime_backend = str(model_cfg.get("runtime_backend", "")).strip().lower()
    # vLLM manages GPU work in child processes; touching torch.cuda in the
    # parent process can break engine startup under fork-based launch.
    _set_generation_seed(generation_seed, touch_torch=(runtime_backend != "vllm"))

    output_root = _resolve_output_root(cfg_path, cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_a_dir = output_root / "stage_a"
    stage_a_dir.mkdir(parents=True, exist_ok=True)
    offline_dir = stage_a_dir / "offline_supervision"

    status = query_gpu_status()
    if (not args.dry_run) and status is not None and is_gpu_busy(
        status,
        max_used_ratio=args.max_gpu_used_ratio,
        max_util_percent=args.max_gpu_util,
    ):
        print(
            (
                "GPU busy, stop now: "
                f"name={status.name}, used={status.used_mib}/{status.total_mib} MiB "
                f"({status.used_ratio:.1%}), util={status.util_percent}%"
            ),
            file=sys.stderr,
        )
        return 2

    items = _load_stage_items(cfg_path, cfg)
    if int(args.max_items_per_dataset) > 0:
        limited: List[StageItem] = []
        counts: Dict[str, int] = {}
        for item in items:
            seen = counts.get(item.dataset, 0)
            if seen >= int(args.max_items_per_dataset):
                continue
            counts[item.dataset] = seen + 1
            limited.append(item)
        items = limited

    normalized_items_rows = [
        {
            "dataset": item.dataset,
            "problem_id": item.problem_id,
            "prompt": item.prompt,
            "reference": item.reference,
            "category": item.category,
            "split": item.split,
            "source_path": item.source_path,
            "source_row_index": item.source_row_index,
        }
        for item in items
    ]
    normalized_items_path = write_jsonl(stage_a_dir / "train_items.normalized.jsonl", normalized_items_rows)

    prompt_template = str(
        cfg.get(
            "prompt_template",
            (
                "You are solving a reasoning task.\n"
                "Dataset: {dataset}\n"
                "Problem ID: {problem_id}\n\n"
                "{prompt}\n\n"
                "Output requirements:\n"
                "- Keep reasoning concise.\n"
                "- For non-code tasks, after </think> output exactly one final line: FINAL_ANSWER: <answer>\n"
                "- For non-code tasks, do not add any explanation before or after that final answer line.\n"
                "- For code tasks, after </think> output FINAL_ANSWER: followed immediately by the final code solution.\n"
            ),
        )
    )
    use_task_metadata_in_prompt = bool(cfg.get("use_task_metadata_in_prompt", False))
    code_policy = str(cfg.get("code_correctness_policy", "collect_prediction_only")).strip() or "collect_prediction_only"
    code_exec_timeout_sec = float(cfg.get("code_exec_timeout_sec", 4.0) or 4.0)
    code_exec_python_bin = str(cfg.get("code_exec_python_bin", "python") or "python")
    on_error = str(cfg.get("on_error", "skip")).strip() or "skip"
    answer_completion_budget = int(cfg.get("answer_completion_budget", 48) or 0)
    boundary_min_tokens = int(cfg.get("boundary_min_tokens", 6) or 6)
    configured_batch_size = max(1, int(cfg.get("inference_batch_size", model_cfg.get("inference_batch_size", 1)) or 1))
    sort_prompts_by_length = bool(cfg.get("sort_prompts_by_length", True))

    stage_a_modes = parse_static_modes(cfg.get("stage_a_modes", cfg.get("static_mode_candidates", [])))
    backend: Any = None
    base_entries: List[Dict[str, Any]] = []
    if args.phase in {"all", "infer"}:
        backend = build_custom_backend(model_cfg, dry_run=args.dry_run)
        base_entries = [
            {
                "item": item,
                "prompt_text": _build_prompt(item, prompt_template, use_task_metadata=use_task_metadata_in_prompt),
            }
            for item in items
        ]
        if sort_prompts_by_length:
            base_entries = sorted(base_entries, key=lambda entry: len(str(entry["prompt_text"])))

    raw_predictions_path = stage_a_dir / "raw_predictions.jsonl"
    scored_raw_rows_path = stage_a_dir / "_scored_raw_rows.jsonl"
    trace_skipped_path = stage_a_dir / "trace_skipped_rows.jsonl"
    static_traces_path = stage_a_dir / "static_traces.jsonl"
    hard_eval_items_path = stage_a_dir / "hard_eval_items.jsonl"
    progress_state_path = stage_a_dir / "_progress_state.json"
    resolved_cfg_path = stage_a_dir / "resolved_config.json"

    if args.no_resume and args.phase != "finalize":
        for path in (
            raw_predictions_path,
            scored_raw_rows_path,
            trace_skipped_path,
            static_traces_path,
            hard_eval_items_path,
            progress_state_path,
            stage_a_dir / "summary.json",
        ):
            _remove_if_exists(path)
        if offline_dir.exists():
            shutil.rmtree(offline_dir)

    completed_keys: set[Tuple[str, str, str]] = set()
    inference_counts = {
        "raw_predictions": 0,
        "scored_raw_rows": 0,
        "prediction_only_rows": 0,
        "eval_error_rows": 0,
    }
    if (not args.no_resume) and raw_predictions_path.exists():
        completed_keys, inference_counts = _scan_existing_raw_rows(raw_predictions_path)

    total_calls = len(items) * len(stage_a_modes)
    done_calls = len(completed_keys)
    supports_batch = bool(backend is not None and hasattr(backend, "generate_batch"))
    print(
        (
            f"[stage_a] items={len(items)} modes={len(stage_a_modes)} "
            f"inference_batch_size={configured_batch_size} supports_batch={supports_batch} "
            f"phase={args.phase} resume={not args.no_resume} completed={done_calls}"
        ),
        flush=True,
    )

    if args.phase in {"all", "infer"}:
        progress = _ProgressPrinter(total=total_calls, done=done_calls)
        for mode in stage_a_modes:
            mode_batch_size = configured_batch_size
            idx = 0
            while idx < len(base_entries):
                batch_n = min(mode_batch_size, len(base_entries) - idx)
                batch_entries = base_entries[idx : idx + batch_n]
                pending_entries = [
                    entry
                    for entry in batch_entries
                    if (
                        entry["item"].dataset,
                        entry["item"].problem_id,
                        mode.tag,
                    )
                    not in completed_keys
                ]
                if not pending_entries:
                    idx += batch_n
                    continue

                batch_items = [entry["item"] for entry in pending_entries]
                prompts = [str(entry["prompt_text"]) for entry in pending_entries]
                answer_trigger_texts = [
                    _answer_trigger_text_for_category(item.category) for item in batch_items
                ]

                call_start = time.perf_counter()
                try:
                    if supports_batch and len(batch_items) > 1:
                        batch_results = backend.generate_batch(  # type: ignore[attr-defined]
                            prompts,
                            info_mode=mode.info_mode,
                            cot_mode=mode.cot_mode,
                            answer_trigger_texts=answer_trigger_texts,
                            answer_completion_budget=answer_completion_budget,
                        )
                    else:
                        batch_results = [
                            backend.generate(
                                prompt,
                                info_mode=mode.info_mode,
                                cot_mode=mode.cot_mode,
                                answer_trigger_text=answer_trigger_texts[j],
                                answer_completion_budget=answer_completion_budget,
                            )
                            for j, prompt in enumerate(prompts)
                        ]
                except Exception as exc:
                    if len(batch_items) > 1 and _is_oom_error(exc):
                        mode_batch_size = max(1, mode_batch_size // 2)
                        _maybe_empty_cuda_cache(backend)
                        print(
                            (
                                f"[warn] OOM for mode=i{mode.info_mode}_c{mode.cot_mode}; "
                                f"retry with batch_size={mode_batch_size}"
                            ),
                            flush=True,
                        )
                        continue
                    raise

                elapsed = max(1e-9, time.perf_counter() - call_start)
                per_item_elapsed = elapsed / max(1, len(batch_items))
                total_batch_tokens = float(sum(int(result.token_count) for result in batch_results))
                batch_tokens_per_sec = total_batch_tokens / elapsed if total_batch_tokens > 0 else 0.0

                for item, result in zip(batch_items, batch_results):
                    latency_ms = per_item_elapsed * 1000.0
                    row: Dict[str, Any] = {
                        "dataset": item.dataset,
                        "problem_id": item.problem_id,
                        "category": item.category,
                        "prompt": item.prompt,
                        "reference": item.reference,
                        "prediction": _normalize_prediction_for_category(item.category, result.prediction),
                        "token_count": int(result.token_count),
                        "info_mode": mode.info_mode,
                        "cot_mode": mode.cot_mode,
                        "mode_tag": mode.tag,
                        "static_config_id": f"info{mode.info_mode}_cot{mode.cot_mode}",
                        "latency_ms": latency_ms,
                        "wall_clock_generation_time_sec": per_item_elapsed,
                        "tokens_per_sec": batch_tokens_per_sec,
                        "generation_seed": generation_seed,
                        "sampling_seed": int(result.raw.get("sampling_seed", generation_seed) or generation_seed),
                        "source_split": item.split,
                        "source_path": item.source_path,
                        "source_row_index": item.source_row_index,
                    }
                    row["temperature_actual"] = result.raw.get("temperature_actual", result.raw.get("temperature"))
                    row["top_p_actual"] = result.raw.get("top_p_actual", result.raw.get("top_p"))
                    row["top_k_actual"] = result.raw.get("top_k_actual")
                    row["prompt_token_count"] = int(result.raw.get("prompt_token_count", 0) or 0)
                    row["vllm_version"] = result.raw.get("vllm_version")
                    row["finish_reason"] = result.raw.get("finish_reason")
                    row["stop_reason"] = result.raw.get("stop_reason")
                    row["stop_token_ids"] = result.raw.get("stop_token_ids")
                    row["backend_raw"] = {k: v for k, v in result.raw.items() if not str(k).startswith("_")}

                    if (
                        str(result.raw.get("backend", "")) == "vllm"
                        and "_token_ids" in result.raw
                        and "_logprobs_list" in result.raw
                    ):
                        tokenizer = getattr(backend, "tokenizer", None)
                        if tokenizer is None:
                            raise ValueError("vLLM post-process requires backend tokenizer.")
                        processed = post_process_vllm_output(
                            prediction=result.prediction,
                            token_ids=result.raw["_token_ids"],
                            logprobs_list=result.raw["_logprobs_list"],
                            tokenizer=tokenizer,
                            config=PostProcessConfig(
                                boundary_min_tokens=boundary_min_tokens,
                                max_tokens=max(
                                    1,
                                    int(result.raw.get("max_new_tokens", result.token_count) or result.token_count),
                                ),
                                info_mode=mode.info_mode,
                                cot_mode=mode.cot_mode,
                                category=item.category,
                                eos_token_id=getattr(tokenizer, "eos_token_id", None),
                            ),
                        )
                        row["mode_trace"] = processed["mode_trace"]
                        row["boundary_states"] = processed["boundary_states"]
                        row["boundary_count"] = int(processed["summary"].get("total_boundaries", 0) or 0)
                        row["has_answer_zone"] = bool(processed["summary"].get("has_answer_zone", False))
                        for key in (
                            "answer_zone_start_token",
                            "think_zone_start_token",
                            "think_zone_end_token",
                            "implicit_think_zone",
                            "think_zone_token_count",
                            "answer_zone_token_count",
                            "prediction_char_length",
                            "prediction_word_length",
                            "n_code_blocks",
                            "n_latex_blocks",
                            "n_numbered_steps",
                        ):
                            if key in processed["summary"]:
                                row[key] = processed["summary"][key]

                    try:
                        if item.category == "code":
                            if code_policy == "collect_prediction_only":
                                row["eval_status"] = "prediction_only"
                            elif code_policy in {"execute_tests", "exec_test", "code_exec"}:
                                if item.dataset == "mbpp" and not item.exec_tests:
                                    raise ValueError(
                                        f"MBPP sample {item.problem_id} is missing executable tests."
                                    )
                                exec_correct, exec_error = _evaluate_code_prediction(
                                    item,
                                    row["prediction"],
                                    timeout_sec=code_exec_timeout_sec,
                                    python_bin=code_exec_python_bin,
                                )
                                row["is_correct"] = float(exec_correct)
                                row["is_correct_exec"] = float(exec_correct)
                                if exec_error:
                                    row["exec_error"] = exec_error
                                row["eval_status"] = "scored"
                            else:
                                row["is_correct"] = infer_correctness(row, code_policy=code_policy)
                                row["eval_status"] = "scored"
                        else:
                            row["is_correct"] = infer_correctness(row, code_policy=code_policy)
                            row["eval_status"] = "scored"
                    except Exception as exc:
                        row["eval_status"] = "eval_error"
                        row["eval_error"] = str(exc)
                        if on_error != "skip":
                            raise

                    _append_jsonl(raw_predictions_path, row)
                    if row.get("eval_status") == "scored":
                        _append_jsonl(scored_raw_rows_path, row)
                        inference_counts["scored_raw_rows"] += 1
                    else:
                        skip_row = {
                            "dataset": row["dataset"],
                            "problem_id": row["problem_id"],
                            "info_mode": row["info_mode"],
                            "cot_mode": row["cot_mode"],
                            "mode_tag": row["mode_tag"],
                            "reason": row.get("eval_status"),
                            "eval_error": row.get("eval_error"),
                        }
                        _append_jsonl(trace_skipped_path, skip_row)
                        if row.get("eval_status") == "prediction_only":
                            inference_counts["prediction_only_rows"] += 1
                        elif row.get("eval_status") == "eval_error":
                            inference_counts["eval_error_rows"] += 1
                    inference_counts["raw_predictions"] += 1
                    completed_keys.add((item.dataset, item.problem_id, mode.tag))
                    done_calls += 1
                    progress.update(
                        1,
                        extra=(
                            f"dataset={item.dataset} problem_id={item.problem_id} "
                            f"mode=i{mode.info_mode}_c{mode.cot_mode} status={row['eval_status']}"
                        ),
                    )
                    progress_state = {
                        "phase": "infer",
                        "done_calls": done_calls,
                        "total_calls": total_calls,
                        "last_dataset": item.dataset,
                        "last_problem_id": item.problem_id,
                        "last_mode_tag": mode.tag,
                    }
                    progress_state_path.write_text(
                        json.dumps(progress_state, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                idx += batch_n
    elif not raw_predictions_path.exists():
        raise FileNotFoundError(
            f"`--phase finalize` requires existing raw_predictions.jsonl at {raw_predictions_path}"
        )

    if args.phase == "infer":
        resolved_cfg_path.write_text(
            json.dumps(
                {
                    "pipeline_config": str(cfg_path),
                    "model_config": model_cfg,
                    "stage_a_modes": [asdict(mode) for mode in stage_a_modes],
                    "dataset_root": str(Path(str(cfg.get("dataset_root", ""))).expanduser()),
                    "datasets": cfg.get("datasets", []),
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "pipeline_config": str(cfg_path),
            "model_config": str(resolve_path(cfg_path, str(cfg.get("model_config")))),
            "dry_run": bool(args.dry_run),
            "stage": "A",
            "generation_seed": generation_seed,
            "code_correctness_policy": code_policy,
            "output_root": str(output_root.resolve()),
            "files": {
                "normalized_items": str(normalized_items_path.resolve()),
                "raw_predictions": str(raw_predictions_path.resolve()),
                "trace_skipped_rows": str(trace_skipped_path.resolve()),
                "resolved_config": str(resolved_cfg_path.resolve()),
            },
            "counts": {
                "normalized_items": len(normalized_items_rows),
                **inference_counts,
            },
            "stage_a_modes": [asdict(mode) for mode in stage_a_modes],
        }
        (stage_a_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not scored_raw_rows_path.exists():
        _remove_if_exists(scored_raw_rows_path)
        _remove_if_exists(trace_skipped_path)
        print("[finalize] rebuilding scored_raw_rows.jsonl from raw_predictions.jsonl", flush=True)
        for row in _iter_jsonl(raw_predictions_path):
            if row.get("eval_status") == "scored":
                _append_jsonl(scored_raw_rows_path, row)
            else:
                skip_row = {
                    "dataset": row.get("dataset"),
                    "problem_id": row.get("problem_id"),
                    "info_mode": row.get("info_mode"),
                    "cot_mode": row.get("cot_mode"),
                    "mode_tag": row.get("mode_tag"),
                    "reason": row.get("eval_status"),
                    "eval_error": row.get("eval_error"),
                }
                _append_jsonl(trace_skipped_path, skip_row)

    print("[finalize] loading scored rows for trace collection", flush=True)
    trace_input_rows = _read_jsonl(scored_raw_rows_path)
    trace_result = collect_static_traces(trace_input_rows, code_policy="weak_text_match", on_error=on_error)
    static_traces_path = write_jsonl(static_traces_path, trace_result.traces)

    print("[finalize] building offline supervision outputs", flush=True)
    offline_cfg = cfg.get("offline_supervision", {}) if isinstance(cfg.get("offline_supervision"), dict) else {}
    pair_cfg = offline_cfg.get("pair_generation", {}) if isinstance(offline_cfg.get("pair_generation"), dict) else {}
    hard_cfg = offline_cfg.get("hard_example", {}) if isinstance(offline_cfg.get("hard_example"), dict) else {}
    offline_summary = _prepare_offline_outputs(
        trace_result.traces,
        output_dir=offline_dir,
        lambda_token_penalty=float(offline_cfg.get("lambda_token_penalty", 0.2)),
        hidden_proj_dim=int(offline_cfg.get("hidden_proj_dim", 0) or 0),
        token_z_threshold=float(hard_cfg.get("token_z_threshold", 1.0)),
        min_score_gap=float(pair_cfg.get("min_score_gap", 0.2)),
        max_pairs_per_problem=int(pair_cfg.get("max_pairs_per_problem", 4)),
        pair_strategy=str(pair_cfg.get("pair_strategy", "multi_pair")),
        disagreement_reweight=bool(pair_cfg.get("disagreement_reweight", True)),
    )
    hard_eval_items = _hard_items_from_normalized_rows(normalized_items_rows, offline_summary.get("hard_examples", []))
    hard_eval_items_path = write_jsonl(stage_a_dir / "hard_eval_items.jsonl", hard_eval_items)

    resolved_cfg_path.write_text(
        json.dumps(
            {
                "pipeline_config": str(cfg_path),
                "model_config": model_cfg,
                "stage_a_modes": [asdict(mode) for mode in stage_a_modes],
                "dataset_root": str(Path(str(cfg.get("dataset_root", ""))).expanduser()),
                "datasets": cfg.get("datasets", []),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_counts = dict(inference_counts)
    if raw_predictions_path.exists():
        _, raw_counts = _scan_existing_raw_rows(raw_predictions_path)
    trace_skipped_count = sum(1 for _ in _iter_jsonl(trace_skipped_path))
    summary = {
        "pipeline_config": str(cfg_path),
        "model_config": str(resolve_path(cfg_path, str(cfg.get("model_config")))),
        "dry_run": bool(args.dry_run),
        "stage": "A",
        "generation_seed": generation_seed,
        "code_correctness_policy": code_policy,
        "output_root": str(output_root.resolve()),
        "files": {
            "normalized_items": str(normalized_items_path.resolve()),
            "raw_predictions": str(raw_predictions_path.resolve()),
            "trace_skipped_rows": str(trace_skipped_path.resolve()),
            "static_traces": str(static_traces_path.resolve()),
            "hard_eval_items": str(hard_eval_items_path.resolve()),
            "resolved_config": str(resolved_cfg_path.resolve()),
            **offline_summary.get("files", {}),
        },
        "counts": {
            "normalized_items": len(items),
            "raw_predictions": raw_counts["raw_predictions"],
            "scored_raw_rows": raw_counts["scored_raw_rows"],
            "prediction_only_rows": raw_counts["prediction_only_rows"],
            "eval_error_rows": raw_counts["eval_error_rows"],
            "trace_skipped_rows": trace_skipped_count,
            "static_traces": len(trace_result.traces),
            "trace_errors": len(trace_result.errors),
            "hard_eval_items": len(hard_eval_items),
            **offline_summary.get("counts", {}),
        },
        "aggregate_raw": _aggregate_raw_from_file(raw_predictions_path),
        "stage_a_modes": [asdict(mode) for mode in stage_a_modes],
    }
    summary_path = stage_a_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
