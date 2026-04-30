from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

from . import math_grading
from .settings import COT_MODE_TABLE, DATASET_REGISTRY, INFO_MODE_TABLE


_RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class CollectionResult:
    traces: List[Dict[str, Any]]
    errors: List[str]


def _canonicalize_number_str(value: str) -> str:
    s = str(value).replace(",", "").strip()
    if not s:
        return ""
    try:
        number = float(s)
    except Exception:
        return s
    if not math.isfinite(number):
        return s
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return str(number)


def _dataset_category(row: Dict[str, Any]) -> str:
    dataset = str(row.get("dataset", "")).strip()
    if dataset in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset].category
    category = str(row.get("category", "")).strip().lower()
    if category:
        return category
    return "unknown"


def _extract_token_count(row: Dict[str, Any]) -> int:
    token_count = row.get("token_count")
    if isinstance(token_count, int) and token_count >= 0:
        return token_count

    usage = row.get("usage", {})
    if isinstance(usage, dict):
        for key in ("completion_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                return value
    raise ValueError(f"Missing token_count for problem_id={row.get('problem_id')}")


def _normalize_choice(text: str) -> str:
    s = text.strip()
    if not s:
        return ""

    final_lines = re.findall(r"(?i)FINAL_?ANSWER\s*:\s*([^\n]+)", s)
    if final_lines:
        tail = final_lines[-1].strip().upper()
        m = re.search(r"^[\(\[]?([A-J])[\)\]\.\,\:\-]?$", tail)
        if m:
            return m.group(1).upper()

    # Prefer explicit answer mentions.
    explicit = re.findall(
        r"(?i)(?:final answer|answer|option|choose)\s*(?:is|:)?\s*[\(\[]?([A-J])[\)\]]?",
        s,
    )
    if explicit:
        return explicit[-1].upper()

    # Next, check if the last non-empty line is just a choice letter.
    for line in reversed(s.splitlines()):
        stripped = line.strip()
        m = re.match(r"^[\(\[]?([A-J])[\)\]\.\,\:\-]?$", stripped, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Fallback: use the last standalone A-J token.
    tokens = re.findall(r"\b([A-J])\b", s.upper())
    if tokens:
        return tokens[-1]
    return s[:1].upper()


def _extract_last_number(text: str) -> str:
    matches = _RE_NUMBER.findall(text.replace(",", ""))
    if not matches:
        return ""
    return _canonicalize_number_str(matches[-1])


def _extract_last_boxed_content(text: str) -> str:
    marker = r"\boxed{"
    start = -1
    search_pos = 0
    while True:
        i = text.find(marker, search_pos)
        if i < 0:
            break
        start = i + len(marker)
        search_pos = i + 1
    if start < 0:
        return ""

    depth = 1
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            out.append(ch)
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
            continue
        out.append(ch)
    return "".join(out).strip()


def _normalize_math_text(text: str) -> str:
    s = text.strip()
    if not s:
        return ""

    final_lines = re.findall(r"(?i)FINAL_?ANSWER\s*:\s*([^\n]+)", s)
    if final_lines:
        s = final_lines[-1].strip()

    # Support GSM8K style "#### 42" first; fall back to last numeric token.
    if "####" in s:
        tail = s.split("####")[-1].strip()
        number = _extract_last_number(tail)
        if number:
            return number
        return re.sub(r"\s+", "", tail).strip("$")

    # MATH often contains boxed final answers.
    boxed = _extract_last_boxed_content(s)
    if boxed:
        return re.sub(r"\s+", "", boxed).strip("$")

    cleaned = re.sub(r"\s+", "", s).strip("$")
    if cleaned and len(cleaned) <= 80 and (("\\" in cleaned) or re.search(r"[A-Za-z]", cleaned)):
        return cleaned

    number = _extract_last_number(s)
    if number:
        return number
    return cleaned


def _extract_math_answer_candidate(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    final_lines = re.findall(r"(?i)FINAL_?ANSWER\s*:\s*([^\n]+)", s)
    if final_lines:
        return final_lines[-1].strip()
    boxed = _extract_last_boxed_content(s)
    if boxed:
        return boxed.strip()
    return s


def _normalize_bool_text(text: str) -> str:
    s = text.strip().lower()
    if not s:
        return ""
    explicit = re.findall(r"(?:final answer|answer|label|classification)\s*(?:is|:)?\s*(true|false|yes|no)", s)
    if explicit:
        token = explicit[-1]
    else:
        tokens = re.findall(r"\b(true|false|yes|no)\b", s)
        if not tokens:
            return ""
        token = tokens[-1]
    return "true" if token in {"true", "yes"} else "false"


def _normalize_code_text(text: str) -> str:
    s = text.strip()
    if not s:
        return ""

    # Prefer fenced code block if present.
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        s = blocks[-1].strip()

    lines = []
    for line in s.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        lines.append(re.sub(r"\s+", " ", stripped.strip()))
    return "\n".join(lines)


def extract_answer_text(
    row: Dict[str, Any],
    *,
    code_policy: str = "require_label",
) -> str:
    prediction = str(row.get("prediction", "")).strip()
    if not prediction:
        return ""
    category = _dataset_category(row)

    if category == "logic":
        normalized_bool = _normalize_bool_text(prediction)
        if normalized_bool:
            return normalized_bool
        return _normalize_choice(prediction)

    if category == "math":
        return _normalize_math_text(prediction)

    if category == "code":
        if code_policy not in {"require_label", "weak_text_match"}:
            return ""
        return _normalize_code_text(prediction)

    return prediction


def _prediction_char_length(text: str) -> int:
    return len(str(text))


def _prediction_word_length(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text), flags=re.UNICODE))


def _prediction_structure_counts(text: str) -> Dict[str, int]:
    content = str(text)
    return {
        "n_code_blocks": len(re.findall(r"```.*?```", content, flags=re.DOTALL)),
        "n_latex_blocks": len(re.findall(r"\$\$.*?\$\$|\$[^$\n]+\$", content, flags=re.DOTALL)),
        "n_numbered_steps": len(
            re.findall(r"(?im)^\s*(?:step\s+\d+\s*[:.\-]|\d+[.)]\s+)", content)
        )
        + len(re.findall(r"(首先|其次|然后|最后)", content)),
    }


def _attach_per_problem_stats(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for trace in traces:
        grouped.setdefault((str(trace["dataset"]), str(trace["problem_id"])), []).append(trace)

    enriched: List[Dict[str, Any]] = []
    for trace in traces:
        rows = grouped[(str(trace["dataset"]), str(trace["problem_id"]))]
        token_counts = [int(item["token_count"]) for item in rows]
        correct_rows = [item for item in rows if float(item.get("correctness", 0.0)) > 0.0]
        correct_token_counts = [int(item["token_count"]) for item in correct_rows]
        stats = {
            "n_configs_tried": len(rows),
            "n_configs_correct": len(correct_rows),
            "min_tokens_among_correct": min(correct_token_counts) if correct_token_counts else None,
            "max_tokens_among_correct": max(correct_token_counts) if correct_token_counts else None,
            "mean_tokens_all_configs": float(mean(token_counts)) if token_counts else 0.0,
            "std_tokens_all_configs": float(pstdev(token_counts)) if len(token_counts) > 1 else 0.0,
            "difficulty_score": 1.0 - (float(len(correct_rows)) / float(max(1, len(rows)))),
        }
        enriched_trace = dict(trace)
        enriched_trace["per_problem_stats"] = stats
        enriched.append(enriched_trace)
    return enriched


def infer_correctness(
    row: Dict[str, Any],
    *,
    code_policy: str = "require_label",
) -> float:
    labeled = row.get("is_correct")
    if isinstance(labeled, bool):
        return 1.0 if labeled else 0.0
    if isinstance(labeled, (int, float)):
        return float(labeled)

    prediction = str(row.get("prediction", "")).strip()
    reference = str(row.get("reference", "")).strip()
    category = _dataset_category(row)

    if not prediction or not reference:
        raise ValueError(
            "Missing prediction/reference and is_correct not provided "
            f"for problem_id={row.get('problem_id')}"
        )

    if category == "logic":
        ref_bool = _normalize_bool_text(reference)
        pred_bool = _normalize_bool_text(prediction)
        if ref_bool:
            return 1.0 if ref_bool == pred_bool else 0.0
        return 1.0 if _normalize_choice(prediction) == _normalize_choice(reference) else 0.0

    if category == "math":
        if str(row.get("dataset", "")).strip() == "math_500" and math_grading.is_available():
            pred_candidate = _extract_math_answer_candidate(prediction)
            ref_candidate = _extract_math_answer_candidate(reference)
            if pred_candidate and ref_candidate:
                return 1.0 if math_grading.grade_answer(pred_candidate, ref_candidate) else 0.0
        pred = _normalize_math_text(prediction)
        ref = _normalize_math_text(reference)
        return 1.0 if pred and ref and pred == ref else 0.0

    if category == "code":
        if code_policy == "weak_text_match":
            pred_norm = _normalize_code_text(prediction)
            ref_norm = _normalize_code_text(reference)
            if not pred_norm or not ref_norm:
                return 0.0
            if pred_norm == ref_norm:
                return 1.0
            # Allow explanation around correct code snippet in baseline stage.
            return 1.0 if ref_norm in pred_norm else 0.0
        raise ValueError(
            "Code sample requires explicit is_correct label "
            f"(problem_id={row.get('problem_id')})"
        )

    # Unknown dataset/category fallback: strict text match.
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def make_trace_id(row: Dict[str, Any], index: int) -> str:
    dataset = str(row.get("dataset", "unknown")).strip() or "unknown"
    problem_id = str(row.get("problem_id", f"idx{index}")).strip() or f"idx{index}"
    info_mode = int(row["info_mode"])
    cot_mode = int(row["cot_mode"])
    return f"{dataset}_{problem_id}_i{info_mode}_c{cot_mode}_{index}"


def to_static_trace(
    row: Dict[str, Any],
    *,
    index: int,
    code_policy: str = "require_label",
) -> Dict[str, Any]:
    dataset = str(row.get("dataset", "")).strip()
    problem_id = str(row.get("problem_id", "")).strip()
    if not dataset:
        raise ValueError("Missing dataset field.")
    if not problem_id:
        raise ValueError("Missing problem_id field.")

    info_mode = int(row["info_mode"])
    cot_mode = int(row["cot_mode"])
    if info_mode not in INFO_MODE_TABLE:
        raise ValueError(f"info_mode out of range: {info_mode}")
    if cot_mode not in COT_MODE_TABLE:
        raise ValueError(f"cot_mode out of range: {cot_mode}")

    correctness = infer_correctness(row, code_policy=code_policy)
    token_count = _extract_token_count(row)

    trace = {
        "trace_id": make_trace_id(row, index),
        "dataset": dataset,
        "problem_id": problem_id,
        "info_mode": info_mode,
        "cot_mode": cot_mode,
        "correctness": float(correctness),
        "token_count": token_count,
        "answer_extracted": str(row.get("answer_extracted", extract_answer_text(row, code_policy=code_policy))),
    }

    prediction = str(row.get("prediction", ""))
    trace["prediction_char_length"] = int(row.get("prediction_char_length", _prediction_char_length(prediction)) or 0)
    trace["prediction_word_length"] = int(row.get("prediction_word_length", _prediction_word_length(prediction)) or 0)
    for key, value in _prediction_structure_counts(prediction).items():
        trace[key] = int(row.get(key, value) or 0)

    # Preserve optional fields that can help later diagnostics.
    for key in (
        "prediction",
        "reference",
        "run_id",
        "metadata",
        "latency_ms",
        "tokens_per_sec",
        "mode_trace",
        "probe_entropy",
        "probe_margin",
        "progress_ratio",
        "output_form",
        "min_new_tokens_before_eos",
        "mode_tag",
        "static_config_id",
        "sampling_seed",
        "prompt_token_count",
        "temperature_actual",
        "top_p_actual",
        "top_k_actual",
        "wall_clock_generation_time_sec",
        "finish_reason",
        "stop_reason",
        "stop_token_ids",
        "boundary_count",
        "has_answer_zone",
        "answer_zone_start_token",
        "think_zone_start_token",
        "think_zone_end_token",
        "implicit_think_zone",
        "think_zone_token_count",
        "answer_zone_token_count",
        "boundary_states",
        "n_code_blocks",
        "n_latex_blocks",
        "n_numbered_steps",
    ):
        if key in row:
            trace[key] = row[key]
    return trace


def collect_static_traces(
    rows: List[Dict[str, Any]],
    *,
    code_policy: str = "require_label",
    on_error: str = "raise",
) -> CollectionResult:
    traces: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, row in enumerate(rows, start=1):
        try:
            traces.append(to_static_trace(row, index=idx, code_policy=code_policy))
        except Exception as exc:  # intentionally aggregate row-level data issues
            msg = f"row={idx} dataset={row.get('dataset')} problem_id={row.get('problem_id')}: {exc}"
            if on_error == "skip":
                errors.append(msg)
                continue
            raise ValueError(msg) from exc
    return CollectionResult(traces=_attach_per_problem_stats(traces), errors=errors)


def summarize_traces(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_dataset: Dict[str, Dict[str, Any]] = {}
    for row in traces:
        ds = str(row["dataset"])
        bucket = per_dataset.setdefault(
            ds,
            {"count": 0, "avg_correctness": 0.0, "avg_tokens": 0.0},
        )
        bucket["count"] += 1
        bucket["avg_correctness"] += float(row["correctness"])
        bucket["avg_tokens"] += float(row["token_count"])

    for ds, bucket in per_dataset.items():
        c = float(bucket["count"])
        if c > 0:
            bucket["avg_correctness"] /= c
            bucket["avg_tokens"] /= c
        per_dataset[ds] = bucket

    return {
        "count": len(traces),
        "datasets": per_dataset,
    }
