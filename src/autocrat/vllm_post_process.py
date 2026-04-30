"""
Post-process vLLM outputs to produce all fields required by datalist.md.

vLLM gives us:
  - prediction text
  - token_ids
  - per-token logprobs (top-20)

This module computes the remaining fields:
  - repeat_ngram_ratio (per token, sliding window)
  - boundary_kind (newline / punct / step_marker / eos / etc.)
  - progress_ratio, remaining_budget_ratio
  - segment_progress_ratio
  - is_answer_zone
  - eos_prob, eos_rank (from logprobs if eos in top-K)
  - boundary_states (aggregated per-boundary records matching datalist spec)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers (mirror custom_backend.py implementations)
# ---------------------------------------------------------------------------

def _repeat_ngram_ratio(token_ids: Sequence[int], *, n: int = 3, window: int = 128) -> float:
    ids = list(int(x) for x in token_ids[-max(1, int(window)):])
    if len(ids) < n:
        return 0.0
    seen: set = set()
    repeats = 0
    total = 0
    for i in range(len(ids) - n + 1):
        total += 1
        ng = tuple(ids[i: i + n])
        if ng in seen:
            repeats += 1
        seen.add(ng)
    return float(repeats) / float(max(1, total))


def _boundary_kind(decoded_piece: str) -> str:
    """Detect boundary type from a decoded token piece."""
    if not decoded_piece:
        return ""
    if "\n" in decoded_piece:
        return "newline"
    stripped = decoded_piece.rstrip()
    if stripped.endswith((".", "?", "!", "。", "？", "！", ":", "：")):
        return "punct"
    low = stripped.lower()
    if "step " in low and ":" in low[-24:]:
        return "step_marker"
    if "reasoning:" in low[-24:]:
        return "step_marker"
    return ""


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def _find_marker_token_index(token_pieces: Sequence[str], marker: str) -> Optional[int]:
    cumulative = ""
    for i, piece in enumerate(token_pieces):
        prev = cumulative
        cumulative += str(piece)
        if marker not in prev and marker in cumulative:
            return i
    return None


def _prediction_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text), flags=re.UNICODE))


def _count_numbered_steps(text: str) -> int:
    english = re.findall(r"(?im)^\s*(?:step\s+\d+\s*[:.\-]|\d+[.)]\s+)", str(text))
    chinese = re.findall(r"(首先|其次|然后|最后)", str(text))
    return len(english) + len(chinese)


# ---------------------------------------------------------------------------
# Per-token feature extraction from vLLM logprobs
# ---------------------------------------------------------------------------

def extract_per_token_logprob_features(
    logprobs_list: List[Any],
    eos_token_id: Optional[int] = None,
    token_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, float]]:
    """Convert vLLM's per-token logprobs into feature dicts.

    Args:
        logprobs_list: comp.logprobs from vLLM output (list of dicts).
        eos_token_id: tokenizer's eos token id for eos_prob/rank extraction.

    Returns:
        List of feature dicts, one per token.
    """
    features: List[Dict[str, float]] = []
    if not logprobs_list:
        return features

    for step_logprobs in logprobs_list:
        if not step_logprobs:
            features.append({
                "entropy": 0.0, "margin": 0.0,
                "top1_prob": 0.0, "top2_prob": 0.0,
                "topk_mass": 0.0,
                "topk_mass_1": 0.0, "topk_mass_2": 0.0, "topk_mass_5": 0.0,
                "topk_mass_10": 0.0, "topk_mass_20": 0.0,
                "tail_mass_beyond_top20": 1.0,
                "renyi_entropy_2": 0.0, "collision_prob": 0.0,
                "effective_vocab_size": 1.0,
                "logprob_gap_1_to_2": None, "logprob_gap_1_to_5": None, "logprob_gap_1_to_10": None,
                "top1_token_id": -1, "top1_logprob": 0.0,
                "chosen_token_rank": None, "chosen_token_logprob": None,
                "eos_prob": 0.0, "eos_rank": 51.0,
            })
            continue

        items = [(int(tid), _logprob_value(lp)) for tid, lp in step_logprobs.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        probs = [math.exp(lp) for _, lp in items]

        top1_prob = probs[0] if len(probs) >= 1 else 0.0
        top2_prob = probs[1] if len(probs) >= 2 else 0.0
        margin = (items[0][1] - items[1][1]) if len(items) >= 2 else 0.0

        def _mass_at(k: int) -> float:
            return float(sum(probs[: max(0, int(k))]))

        topk_mass_1 = _mass_at(1)
        topk_mass_2 = _mass_at(2)
        topk_mass_5 = _mass_at(5)
        topk_mass_10 = _mass_at(10)
        topk_mass_20 = _mass_at(20)

        entropy = 0.0
        collision_prob = 0.0
        for p in probs:
            if p > 1e-12:
                entropy -= p * math.log(p)
                collision_prob += p * p
        renyi_entropy_2 = -math.log(max(collision_prob, 1e-12)) if collision_prob > 0.0 else 0.0

        eos_prob = 0.0
        eos_rank = float(len(items) + 1)
        if eos_token_id is not None:
            for rank, (tid, lp) in enumerate(items):
                if tid == eos_token_id:
                    eos_prob = math.exp(lp)
                    eos_rank = float(rank + 1)
                    break

        chosen_token_rank: Optional[int] = None
        chosen_token_logprob: Optional[float] = None
        if token_ids is not None and len(token_ids) > len(features):
            chosen_tid = int(token_ids[len(features)])
            for rank, (tid, lp) in enumerate(items):
                if tid == chosen_tid:
                    chosen_token_rank = rank + 1
                    chosen_token_logprob = lp
                    break

        def _logprob_gap_to(k: int) -> Optional[float]:
            if len(items) < max(2, int(k)):
                return None
            return float(items[0][1] - items[int(k) - 1][1])

        features.append({
            "entropy": entropy,
            "margin": margin,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            # Keep `topk_mass` as a scalar alias for top-5 mass so the training
            # pipeline can keep consuming a single field without schema churn.
            "topk_mass": topk_mass_5,
            "topk_mass_1": topk_mass_1,
            "topk_mass_2": topk_mass_2,
            "topk_mass_5": topk_mass_5,
            "topk_mass_10": topk_mass_10,
            "topk_mass_20": topk_mass_20,
            "tail_mass_beyond_top20": max(0.0, 1.0 - topk_mass_20),
            "renyi_entropy_2": renyi_entropy_2,
            "collision_prob": collision_prob,
            "effective_vocab_size": math.exp(entropy),
            "logprob_gap_1_to_2": _logprob_gap_to(2),
            "logprob_gap_1_to_5": _logprob_gap_to(5),
            "logprob_gap_1_to_10": _logprob_gap_to(10),
            "top1_token_id": items[0][0] if items else -1,
            "top1_logprob": items[0][1] if items else 0.0,
            "chosen_token_rank": chosen_token_rank,
            "chosen_token_logprob": chosen_token_logprob,
            "eos_prob": eos_prob,
            "eos_rank": eos_rank,
        })
    return features


# ---------------------------------------------------------------------------
# Full post-processing: produce boundary_states matching datalist.md
# ---------------------------------------------------------------------------

@dataclass
class PostProcessConfig:
    """Configuration for vLLM output post-processing."""
    boundary_min_tokens: int = 6
    max_tokens: int = 512
    info_mode: int = 3
    cot_mode: int = 2
    category: str = ""
    eos_token_id: Optional[int] = None


def post_process_vllm_output(
    *,
    prediction: str,
    token_ids: Sequence[int],
    logprobs_list: List[Any],
    tokenizer: Any,
    config: PostProcessConfig,
) -> Dict[str, Any]:
    """Post-process a single vLLM generation output into datalist-complete format.

    Returns a dict with:
      - token_features: per-token logprob features
      - boundary_states: per-boundary aggregated states (main training data)
      - summary: trajectory-level metadata
    """
    n_tokens = len(token_ids)
    eos_token_id = config.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # 1. Per-token logprob features
    token_features = extract_per_token_logprob_features(logprobs_list, eos_token_id, token_ids=token_ids)

    # 2. Decode each token piece for boundary detection
    token_pieces: List[str] = []
    for tid in token_ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
        token_pieces.append(str(piece))

    # 3. Detect answer / think zones from rendered text markers.
    answer_zone_start = _find_marker_token_index(token_pieces, "</think>")
    if answer_zone_start is None:
        answer_zone_start = _find_marker_token_index(
            token_pieces,
            "FINAL_ANSWER:",
        )
        if answer_zone_start is None:
            answer_zone_start = _find_marker_token_index(
                [piece.upper() for piece in token_pieces],
                "FINAL_ANSWER:",
            )
    think_zone_start = _find_marker_token_index(token_pieces, "<think>")
    think_zone_end = _find_marker_token_index(token_pieces, "</think>")
    implicit_think_zone = False
    if think_zone_end is not None and think_zone_start is None:
        # Some reasoning models emit only the closing tag, meaning "thinking"
        # started from the beginning of generation and ended at </think>.
        think_zone_start = 0
        implicit_think_zone = True

    prediction_text = str(prediction or "")
    prediction_char_length = len(prediction_text)
    prediction_word_length = _prediction_word_count(prediction_text)
    n_code_blocks = len(re.findall(r"```.*?```", prediction_text, flags=re.DOTALL))
    n_latex_blocks = len(re.findall(r"\$\$.*?\$\$|\$[^$\n]+\$", prediction_text, flags=re.DOTALL))
    n_numbered_steps = _count_numbered_steps(prediction_text)
    think_zone_token_count = 0
    if think_zone_start is not None:
        think_zone_stop = think_zone_end if think_zone_end is not None else n_tokens
        think_zone_token_count = max(0, int(think_zone_stop) - int(think_zone_start))
    answer_zone_token_count = 0
    if answer_zone_start is not None:
        answer_zone_token_count = max(0, n_tokens - int(answer_zone_start))

    # 4. Walk through tokens, detect boundaries, build boundary_states
    boundary_states: List[Dict[str, Any]] = []
    tokens_since_boundary = 0
    decoded_tail = ""  # rolling window for boundary detection
    cumulative_entropy_sum = 0.0
    cumulative_margin_sum = 0.0
    _SCALAR_OBS_KEYS = (
        "entropy",
        "margin",
        "top1_prob",
        "top2_prob",
        "topk_mass",
        "topk_mass_20",
        "eos_prob",
        "eos_rank",
        "renyi_entropy_2",
        "effective_vocab_size",
        "tail_mass_beyond_top20",
    )
    segment_obs: Dict[str, List[float]] = {k: [] for k in _SCALAR_OBS_KEYS}

    for i in range(n_tokens):
        piece = token_pieces[i]
        decoded_tail = (decoded_tail + piece)[-128:]
        tokens_since_boundary += 1

        # Accumulate per-token scalar features into segment
        if i < len(token_features):
            tf = token_features[i]
            for key in _SCALAR_OBS_KEYS:
                segment_obs[key].append(float(tf.get(key, 0.0)))
            cumulative_entropy_sum += float(tf.get("entropy", 0.0))
            cumulative_margin_sum += float(tf.get("margin", 0.0))

        # Check if this token is EOS
        is_eos = (eos_token_id is not None and int(token_ids[i]) == eos_token_id)

        # Detect boundary
        bk = _boundary_kind(decoded_tail)
        is_boundary = is_eos or (
            tokens_since_boundary >= config.boundary_min_tokens and bool(bk)
        )

        if not is_boundary and i == n_tokens - 1:
            # Force boundary at end of sequence
            is_boundary = True
            bk = "eos" if is_eos else "end_of_sequence"

        if is_boundary:
            if is_eos:
                bk = "eos"

            # Aggregate segment observables
            def _agg(vals: List[float]) -> Dict[str, float]:
                if not vals:
                    return {"mean": 0.0, "last": 0.0, "slope": 0.0}
                mean = sum(vals) / len(vals)
                last = vals[-1]
                slope = (vals[-1] - vals[0]) / max(1, len(vals) - 1) if len(vals) > 1 else 0.0
                return {"mean": mean, "last": last, "slope": slope}

            agg = {k: _agg(vs) for k, vs in segment_obs.items()}

            # Current token's features (last in segment)
            cur = token_features[i] if i < len(token_features) else {}

            # Compute datalist fields
            progress_ratio = min(1.0, float(i + 1) / float(max(1, config.max_tokens)))
            remaining_budget_ratio = max(0.0, 1.0 - progress_ratio)
            segment_progress_ratio = min(
                1.0,
                float(tokens_since_boundary) / float(max(1, config.boundary_min_tokens * 4)),
            )
            is_answer_zone = (answer_zone_start is not None and i >= answer_zone_start)
            is_code_mode = (config.category.strip().lower() == "code")
            is_in_think_zone = (
                think_zone_start is not None
                and i >= think_zone_start
                and (think_zone_end is None or i < think_zone_end)
            )
            think_zone_depth = (i - think_zone_start) if is_in_think_zone and think_zone_start is not None else -1
            distance_to_answer_zone = -1
            if answer_zone_start is not None and i >= answer_zone_start:
                distance_to_answer_zone = i - answer_zone_start
            segment_index = len(boundary_states)
            top1_token_id = int(cur.get("top1_token_id", -1) or -1)
            top1_token_str = ""
            if top1_token_id >= 0:
                top1_token_str = str(tokenizer.decode([top1_token_id], skip_special_tokens=False))
            window_start = max(0, i - 31)
            recent_window = token_features[window_start : i + 1]
            recent_window_entropy_mean = 0.0
            if recent_window:
                recent_window_entropy_mean = sum(
                    float(item.get("entropy", 0.0)) for item in recent_window
                ) / float(len(recent_window))

            boundary_state = {
                # Logits-derived (datalist: 边界级观测元数据)
                "entropy": cur.get("entropy", 0.0),
                "margin": cur.get("margin", 0.0),
                "top1_prob": cur.get("top1_prob", 0.0),
                "top2_prob": cur.get("top2_prob", 0.0),
                "topk_mass": cur.get("topk_mass", 0.0),
                "topk_mass_1": cur.get("topk_mass_1", 0.0),
                "topk_mass_2": cur.get("topk_mass_2", 0.0),
                "topk_mass_5": cur.get("topk_mass_5", 0.0),
                "topk_mass_10": cur.get("topk_mass_10", 0.0),
                "topk_mass_20": cur.get("topk_mass_20", 0.0),
                "tail_mass_beyond_top20": cur.get("tail_mass_beyond_top20", 1.0),
                "renyi_entropy_2": cur.get("renyi_entropy_2", 0.0),
                "collision_prob": cur.get("collision_prob", 0.0),
                "effective_vocab_size": cur.get("effective_vocab_size", 1.0),
                "logprob_gap_1_to_2": cur.get("logprob_gap_1_to_2"),
                "logprob_gap_1_to_5": cur.get("logprob_gap_1_to_5"),
                "logprob_gap_1_to_10": cur.get("logprob_gap_1_to_10"),
                "chosen_token_rank": cur.get("chosen_token_rank"),
                "chosen_token_logprob": cur.get("chosen_token_logprob"),
                "top1_token_id": top1_token_id,
                "top1_token_str": top1_token_str,
                "eos_prob": cur.get("eos_prob", 0.0),
                "eos_rank": cur.get("eos_rank", 51.0),
                "repeat_ngram_ratio": _repeat_ngram_ratio(
                    token_ids[: i + 1], n=3, window=128
                ),
                # Governance context (datalist: 治理上下文元数据)
                "current_info_mode": config.info_mode,
                "current_cot_mode": config.cot_mode,
                "generated_tokens": i + 1,
                "progress_ratio": progress_ratio,
                "remaining_budget_ratio": remaining_budget_ratio,
                "segment_progress_ratio": segment_progress_ratio,
                "tokens_since_last_boundary": tokens_since_boundary,
                "segment_index": segment_index,
                "distance_to_answer_zone": distance_to_answer_zone,
                "is_in_think_zone": is_in_think_zone,
                "implicit_think_zone": implicit_think_zone,
                "think_zone_depth": think_zone_depth,
                "cumulative_entropy_mean_so_far": cumulative_entropy_sum / float(i + 1),
                "cumulative_margin_mean_so_far": cumulative_margin_sum / float(i + 1),
                "recent_window_entropy_mean": recent_window_entropy_mean,
                "boundary_kind": bk,
                "is_answer_zone": is_answer_zone,
                "is_code_mode": is_code_mode,
                # Segment-level aggregated observables
                "boundary_observables": agg,
                # Mode trace entry (datalist: 模式轨迹元数据)
                "token_index": i,
                "chosen_info_mode": config.info_mode,
                "chosen_cot_mode": config.cot_mode,
                "info_mode_switch_flag": False,
                "cot_mode_switch_flag": False,
                "info_mode_duration": segment_index + 1,
                "cot_mode_duration": segment_index + 1,
            }
            boundary_states.append(boundary_state)

            # Reset segment
            tokens_since_boundary = 0
            decoded_tail = ""
            segment_obs = {k: [] for k in segment_obs}

    # 5. Build mode_trace (datalist: 模式轨迹元数据)
    mode_trace = [
        {
            "token_index": bs["token_index"],
            "chosen_info_mode": bs["chosen_info_mode"],
            "chosen_cot_mode": bs["chosen_cot_mode"],
        }
        for bs in boundary_states
    ]

    return {
        "token_features": token_features,
        "boundary_states": boundary_states,
        "mode_trace": mode_trace,
        "summary": {
            "total_tokens": n_tokens,
            "total_boundaries": len(boundary_states),
            "has_answer_zone": answer_zone_start is not None,
            "answer_zone_start_token": answer_zone_start,
            "think_zone_start_token": think_zone_start,
            "think_zone_end_token": think_zone_end,
            "implicit_think_zone": implicit_think_zone,
            "think_zone_token_count": think_zone_token_count,
            "answer_zone_token_count": answer_zone_token_count,
            "prediction_char_length": prediction_char_length,
            "prediction_word_length": prediction_word_length,
            "n_code_blocks": n_code_blocks,
            "n_latex_blocks": n_latex_blocks,
            "n_numbered_steps": n_numbered_steps,
        },
    }


# ---------------------------------------------------------------------------
# Datalist completeness check
# ---------------------------------------------------------------------------

# All fields required by datalist.md
DATALIST_TRAJECTORY_FIELDS = {
    "dataset", "problem_id", "category", "prompt",
    "info_mode", "cot_mode", "prediction", "correctness", "token_count",
}

DATALIST_BOUNDARY_FIELDS = {
    # 边界级观测元数据
    "entropy", "margin", "top1_prob", "top2_prob",
    "topk_mass", "eos_prob", "eos_rank", "repeat_ngram_ratio",
    # 治理上下文元数据
    "current_info_mode", "current_cot_mode", "generated_tokens",
    "progress_ratio", "remaining_budget_ratio", "segment_progress_ratio",
    "boundary_kind", "is_answer_zone", "is_code_mode",
    # 模式轨迹元数据
    "token_index", "chosen_info_mode", "chosen_cot_mode",
}

DATALIST_MODE_TRACE_FIELDS = {
    "token_index", "chosen_info_mode", "chosen_cot_mode",
}


def check_datalist_completeness(
    boundary_states: List[Dict[str, Any]],
    mode_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check if all datalist.md required fields are present."""
    boundary_missing: List[str] = []
    trace_missing: List[str] = []

    if boundary_states:
        first = boundary_states[0]
        boundary_missing = sorted(DATALIST_BOUNDARY_FIELDS - set(first.keys()))

    if mode_trace:
        first = mode_trace[0]
        trace_missing = sorted(DATALIST_MODE_TRACE_FIELDS - set(first.keys()))

    return {
        "boundary_fields_ok": len(boundary_missing) == 0,
        "boundary_missing": boundary_missing,
        "trace_fields_ok": len(trace_missing) == 0,
        "trace_missing": trace_missing,
        "all_ok": len(boundary_missing) == 0 and len(trace_missing) == 0,
    }
