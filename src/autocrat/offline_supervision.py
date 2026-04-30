from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple


from .settings import DATASET_REGISTRY


@dataclass(frozen=True)
class TokenStats:
    mean_tokens: float
    std_tokens: float


def infer_context_tag(row: Dict[str, Any]) -> str:
    category = str(row.get("category", "")).strip().lower()
    if category == "code":
        return "code"
    dataset = str(row.get("dataset", "")).strip().lower()
    spec = DATASET_REGISTRY.get(dataset)
    if spec is not None and str(spec.category).strip().lower() == "code":
        return "code"
    return "non_code"


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"JSONL file not found: {src}")
    rows: List[Dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} is not a JSON object: {src}")
            rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return dst


def validate_trace_record(row: Dict[str, Any]) -> None:
    required = [
        "trace_id",
        "dataset",
        "problem_id",
        "info_mode",
        "cot_mode",
        "correctness",
        "token_count",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Trace missing required keys: {missing}")
    if not isinstance(row["token_count"], int) or row["token_count"] < 0:
        raise ValueError(f"Invalid token_count in trace_id={row.get('trace_id')}")
    if not isinstance(row["correctness"], (int, float)):
        raise ValueError(f"Invalid correctness in trace_id={row.get('trace_id')}")


def _problem_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row["dataset"]), str(row["problem_id"])


def _boundary_problem_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return str(row["dataset"]), str(row["problem_id"]), int(row["boundary_index"])


def compute_token_stats_per_problem(traces: List[Dict[str, Any]]) -> Dict[Tuple[str, str], TokenStats]:
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for row in traces:
        validate_trace_record(row)
        buckets.setdefault(_problem_key(row), []).append(int(row["token_count"]))

    stats: Dict[Tuple[str, str], TokenStats] = {}
    for key, values in buckets.items():
        m = float(mean(values))
        std = float(pstdev(values))
        stats[key] = TokenStats(mean_tokens=m, std_tokens=(std if std > 1e-9 else 1.0))
    return stats


def score_traces(
    traces: List[Dict[str, Any]],
    lambda_token_penalty: float,
) -> List[Dict[str, Any]]:
    stats = compute_token_stats_per_problem(traces)
    scored: List[Dict[str, Any]] = []
    for row in traces:
        key = _problem_key(row)
        s = stats[key]
        token_z = (int(row["token_count"]) - s.mean_tokens) / s.std_tokens
        score = float(row["correctness"]) - (lambda_token_penalty * token_z)
        enriched = dict(row)
        enriched["token_z"] = token_z
        enriched["score"] = score
        scored.append(enriched)
    return scored


def find_hard_examples(
    scored_traces: List[Dict[str, Any]],
    token_z_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in scored_traces:
        grouped.setdefault(_problem_key(row), []).append(row)

    hard: List[Dict[str, Any]] = []
    for (dataset, problem_id), rows in grouped.items():
        all_incorrect = all(float(item["correctness"]) <= 0.0 for item in rows)
        has_token_outlier = any(float(item["token_z"]) >= token_z_threshold for item in rows)
        if not (all_incorrect or has_token_outlier):
            continue
        reasons = []
        if all_incorrect:
            reasons.append("all_incorrect")
        if has_token_outlier:
            reasons.append("token_outlier")
        hard.append(
            {
                "dataset": dataset,
                "problem_id": problem_id,
                "reasons": reasons,
                "num_traces": len(rows),
            }
        )
    return hard


def build_preference_pairs(
    scored_traces: List[Dict[str, Any]],
    min_score_gap: float = 0.2,
    max_pairs_per_problem: int = 4,
    pair_strategy: str = "multi_pair",
    disagreement_reweight: bool = True,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in scored_traces:
        grouped.setdefault(_problem_key(row), []).append(row)

    pairs: List[Dict[str, Any]] = []
    for (dataset, problem_id), rows in grouped.items():
        ranked = sorted(rows, key=lambda item: float(item["score"]), reverse=True)
        local_pairs: List[Dict[str, Any]] = []
        max_score = float(ranked[0]["score"])
        min_score = float(ranked[-1]["score"])
        spread = max(1e-6, max_score - min_score)
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                winner = ranked[i]
                loser = ranked[j]
                gap = float(winner["score"]) - float(loser["score"])
                if gap < min_score_gap:
                    continue
                local_pairs.append(
                    {
                        "dataset": dataset,
                        "problem_id": problem_id,
                        "winner_trace_id": winner["trace_id"],
                        "loser_trace_id": loser["trace_id"],
                        "score_gap": gap,
                        "winner_mode": {
                            "info_mode": int(winner["info_mode"]),
                            "cot_mode": int(winner["cot_mode"]),
                        },
                        "loser_mode": {
                            "info_mode": int(loser["info_mode"]),
                            "cot_mode": int(loser["cot_mode"]),
                        },
                        "pair_weight": (gap / spread) if disagreement_reweight else 1.0,
                    }
                )
        if pair_strategy == "best_vs_worst":
            if local_pairs:
                local_pairs = [max(local_pairs, key=lambda x: float(x["score_gap"]))]
        else:
            local_pairs = sorted(local_pairs, key=lambda x: float(x["score_gap"]), reverse=True)[
                : max(1, int(max_pairs_per_problem))
            ]
        pairs.extend(local_pairs)
    return pairs


def _coerce_topk_mass(value: Any, *, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if not value:
            return float(fallback)
        # Older trace converters may store cumulative probability-mass lists.
        if len(value) >= 5:
            return float(value[4])
        return float(value[-1])
    return float(fallback)


def _base_state_features(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": str(row.get("prompt", "")),
        "context_tag": infer_context_tag(row),
        "entropy": float(row.get("probe_entropy", row.get("entropy", 0.0)) or 0.0),
        "margin": float(row.get("probe_margin", row.get("margin", 0.0)) or 0.0),
        "top1_prob": float(row.get("probe_top1_prob", row.get("top1_prob", 0.0)) or 0.0),
        "top2_prob": float(row.get("probe_top2_prob", row.get("top2_prob", 0.0)) or 0.0),
        "topk_mass": _coerce_topk_mass(
            row.get("probe_topk_mass", row.get("topk_mass_5", row.get("topk_mass", 0.0))),
            fallback=0.0,
        ),
        "eos_prob": float(row.get("probe_eos_prob", row.get("eos_prob", 0.0)) or 0.0),
        "eos_rank": float(row.get("probe_eos_rank", row.get("eos_rank", 0.0)) or 0.0),
        "repeat_ngram_ratio": float(row.get("probe_repeat_ngram_ratio", row.get("repeat_ngram_ratio", 0.0)) or 0.0),
        "prompt_len": len(str(row.get("prompt", ""))),
        "generated_tokens": int(row.get("token_count", 0) or 0),
        "progress_ratio": float(row.get("progress_ratio", 1.0) or 1.0),
        "current_info_mode": int(row.get("info_mode", 0) or 0),
        "current_cot_mode": int(row.get("cot_mode", 0) or 0),
        "remaining_budget_ratio": float(row.get("remaining_budget_ratio", 1.0) or 1.0),
        "segment_progress_ratio": float(row.get("segment_progress_ratio", 1.0) or 1.0),
        "is_answer_zone": bool(row.get("is_answer_zone", False)),
        "is_code_mode": bool(infer_context_tag(row) == "code"),
        "boundary_kind": str(row.get("boundary_kind", "none") or "none"),
    }


def _boundary_samples_from_row(
    row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    boundary_states = row.get("boundary_states")
    if isinstance(boundary_states, list) and boundary_states:
        samples: List[Dict[str, Any]] = []
        final_score = float(row["score"])
        for idx, event in enumerate(boundary_states):
            if not isinstance(event, dict):
                continue
            state = dict(event)
            state["prompt"] = str(row.get("prompt", ""))
            state["context_tag"] = infer_context_tag(row)
            state["topk_mass"] = _coerce_topk_mass(
                state.get("topk_mass_5", state.get("topk_mass", 0.0)),
                fallback=0.0,
            )
            state["current_info_mode"] = int(state.get("current_info_mode", row.get("info_mode", 0)) or 0)
            state["current_cot_mode"] = int(state.get("current_cot_mode", row.get("cot_mode", 0)) or 0)
            samples.append(
                {
                    "trace_id": str(row["trace_id"]),
                    "dataset": str(row["dataset"]),
                    "problem_id": str(row["problem_id"]),
                    "category": str(row.get("category", "")),
                    "prompt": str(row.get("prompt", "")),
                    "boundary_index": idx,
                    "chosen_info_mode": int(event.get("chosen_info_mode", row.get("info_mode", 0)) or 0),
                    "chosen_cot_mode": int(event.get("chosen_cot_mode", row.get("cot_mode", 0)) or 0),
                    "trajectory_score": final_score,
                    "score": final_score,
                    "token_count": int(row.get("token_count", 0) or 0),
                    "remaining_budget_ratio": float(state.get("remaining_budget_ratio", 1.0) or 1.0),
                    "boundary_kind": str(state.get("boundary_kind", "none") or "none"),
                    "state_features": state,
                }
            )
        if samples:
            return samples

    mode_trace = row.get("mode_trace")
    if not isinstance(mode_trace, list) or not mode_trace:
        state = _base_state_features(row)
        state["boundary_kind"] = str(state.get("boundary_kind", "initial") or "initial")
        state["progress_ratio"] = min(1.0, max(0.0, float(state["progress_ratio"])))
        return [
            {
                "trace_id": str(row["trace_id"]),
                "dataset": str(row["dataset"]),
                "problem_id": str(row["problem_id"]),
                "category": str(row.get("category", "")),
                "prompt": str(row.get("prompt", "")),
                "boundary_index": 0,
                "chosen_info_mode": int(row["info_mode"]),
                "chosen_cot_mode": int(row["cot_mode"]),
                "trajectory_score": float(row["score"]),
                "score": float(row["score"]),
                "token_count": int(row.get("token_count", 0) or 0),
                "remaining_budget_ratio": float(state["remaining_budget_ratio"]),
                "boundary_kind": str(state["boundary_kind"]),
                "state_features": state,
            }
        ]

    samples: List[Dict[str, Any]] = []
    final_score = float(row["score"])
    for idx, event in enumerate(mode_trace):
        if not isinstance(event, dict):
            continue
        extra = event.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        state = {
            "prompt": str(row.get("prompt", "")),
            "context_tag": infer_context_tag(row),
            "entropy": float(event.get("entropy", row.get("probe_entropy", 0.0)) or 0.0),
            "margin": float(event.get("margin", row.get("probe_margin", 0.0)) or 0.0),
            "top1_prob": float(event.get("top1_prob", row.get("probe_top1_prob", row.get("top1_prob", 0.0))) or 0.0),
            "top2_prob": float(event.get("top2_prob", row.get("probe_top2_prob", row.get("top2_prob", 0.0))) or 0.0),
            "topk_mass": _coerce_topk_mass(
                event.get("topk_mass_5", event.get("topk_mass", row.get("probe_topk_mass", row.get("topk_mass", 0.0)))),
                fallback=0.0,
            ),
            "eos_prob": float(event.get("eos_prob", row.get("probe_eos_prob", row.get("eos_prob", 0.0))) or 0.0),
            "eos_rank": float(event.get("eos_rank", row.get("probe_eos_rank", row.get("eos_rank", 0.0))) or 0.0),
            "repeat_ngram_ratio": float(
                event.get("repeat_ngram_ratio", row.get("probe_repeat_ngram_ratio", row.get("repeat_ngram_ratio", 0.0)))
                or 0.0
            ),
            "prompt_len": len(str(row.get("prompt", ""))),
            "generated_tokens": int(event.get("token_index", row.get("token_count", 0)) or 0),
            "progress_ratio": float(extra.get("progress_ratio", row.get("progress_ratio", 1.0) or 1.0)),
            "current_info_mode": int(event.get("info_mode", row.get("info_mode", 0)) or 0),
            "current_cot_mode": int(event.get("cot_mode", row.get("cot_mode", 0)) or 0),
            "remaining_budget_ratio": max(
                0.0,
                min(
                    1.0,
                    1.0 - float(extra.get("progress_ratio", row.get("progress_ratio", 1.0) or 1.0)),
                ),
            ),
            "segment_progress_ratio": float(event.get("segment_progress_ratio", extra.get("segment_progress_ratio", 0.0) or 0.0)),
            "is_answer_zone": bool(extra.get("forced_stop", False) or row.get("is_answer_zone", False)),
            "is_code_mode": bool(infer_context_tag(row) == "code"),
            "boundary_kind": str(event.get("boundary_kind", extra.get("boundary_kind", "none")) or "none"),
        }
        samples.append(
            {
                "trace_id": str(row["trace_id"]),
                "dataset": str(row["dataset"]),
                "problem_id": str(row["problem_id"]),
                "category": str(row.get("category", "")),
                "prompt": str(row.get("prompt", "")),
                "boundary_index": idx,
                "chosen_info_mode": int(event.get("info_mode", row.get("info_mode", 0)) or 0),
                "chosen_cot_mode": int(event.get("cot_mode", row.get("cot_mode", 0)) or 0),
                "trajectory_score": final_score,
                "score": final_score,
                "token_count": int(row.get("token_count", 0) or 0),
                "remaining_budget_ratio": float(state["remaining_budget_ratio"]),
                "boundary_kind": str(state["boundary_kind"]),
                "state_features": state,
            }
        )
    if not samples:
        return _boundary_samples_from_row(dict(row, mode_trace=[]))
    return samples


def build_boundary_samples(
    scored_traces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in scored_traces:
        rows.extend(_boundary_samples_from_row(row))
    return rows


def build_boundary_preference_pairs(
    boundary_samples: List[Dict[str, Any]],
    *,
    min_score_gap: float = 0.2,
    max_pairs_per_problem: int = 4,
    pair_strategy: str = "multi_pair",
    disagreement_reweight: bool = True,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in boundary_samples:
        grouped.setdefault(_boundary_problem_key(row), []).append(row)

    pairs: List[Dict[str, Any]] = []
    for (dataset, problem_id, boundary_index), rows in grouped.items():
        ranked = sorted(rows, key=lambda item: float(item["score"]), reverse=True)
        local_pairs: List[Dict[str, Any]] = []
        max_score = float(ranked[0]["score"])
        min_score = float(ranked[-1]["score"])
        spread = max(1e-6, max_score - min_score)
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                winner = ranked[i]
                loser = ranked[j]
                gap = float(winner["score"]) - float(loser["score"])
                if gap < min_score_gap:
                    continue
                local_pairs.append(
                    {
                        "dataset": dataset,
                        "problem_id": problem_id,
                        "boundary_index": boundary_index,
                        "winner_boundary_id": f"{winner['trace_id']}#{boundary_index}",
                        "loser_boundary_id": f"{loser['trace_id']}#{boundary_index}",
                        "winner_trace_id": winner["trace_id"],
                        "loser_trace_id": loser["trace_id"],
                        "score_gap": gap,
                        "winner_mode": {
                            "info_mode": int(winner["chosen_info_mode"]),
                            "cot_mode": int(winner["chosen_cot_mode"]),
                        },
                        "loser_mode": {
                            "info_mode": int(loser["chosen_info_mode"]),
                            "cot_mode": int(loser["chosen_cot_mode"]),
                        },
                        "pair_weight": (gap / spread) if disagreement_reweight else 1.0,
                    }
                )
        if pair_strategy == "best_vs_worst":
            if local_pairs:
                local_pairs = [max(local_pairs, key=lambda x: float(x["score_gap"]))]
        else:
            local_pairs = sorted(local_pairs, key=lambda x: float(x["score_gap"]), reverse=True)[
                : max(1, int(max_pairs_per_problem))
            ]
        pairs.extend(local_pairs)
    return pairs
