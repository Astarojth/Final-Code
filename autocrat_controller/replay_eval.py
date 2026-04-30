#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT.parent))
    from autocrat_controller.action_space import JointActionSpace
    from autocrat_controller.features import BoundaryFeatureSpec, HashTextVectorizer
    from autocrat_controller.io_utils import read_jsonl, write_json, write_jsonl
    from autocrat_controller.models import MLP
    from autocrat_controller.neighborhood import BoundaryNeighborhoodConfig, BoundaryNeighborhoodSpec, NeighborhoodFeature
    from autocrat_controller.policy import OnlineDecisionPolicy
    from autocrat_controller.slot_memory import SlotMemory
else:
    from .action_space import JointActionSpace
    from .features import BoundaryFeatureSpec, HashTextVectorizer
    from .io_utils import read_jsonl, write_json, write_jsonl
    from .models import MLP
    from .neighborhood import BoundaryNeighborhoodConfig, BoundaryNeighborhoodSpec, NeighborhoodFeature
    from .policy import OnlineDecisionPolicy
    from .slot_memory import SlotMemory


@dataclass(frozen=True)
class ProblemTokenStats:
    mean: float
    std: float


@dataclass(frozen=True)
class TraceRecord:
    dataset: str
    problem_id: str
    prompt: str
    info_mode: int
    cot_mode: int
    score: float
    correctness: float
    token_count: int
    think_tokens: int
    answer_tokens: int
    raw: Dict[str, Any]


@dataclass(frozen=True)
class ProblemStaticState:
    prompt: str
    traces_by_action: Dict[Tuple[int, int], TraceRecord]
    boundaries_by_action: Dict[Tuple[int, int], List[Dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline replay/eval for the AutoCRAT boundary controller.")
    parser.add_argument("--supervision-dir", required=True, help="Path to the offline supervision root.")
    parser.add_argument("--artifact", required=True, help="Path to autocrat_controller.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory to write replay summaries.")
    parser.add_argument("--split", choices=("all", "train", "val"), default="val")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--val-problem-frac", type=float, default=None)
    parser.add_argument("--disable-val", action="store_true")
    parser.add_argument("--prior-weight", type=float, default=0.15)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--switch-cost", type=float, default=0.10)
    parser.add_argument("--hysteresis-bonus", type=float, default=0.05)
    parser.add_argument("--budget-guardrail-penalty", type=float, default=0.35)
    parser.add_argument("--max-switches", type=int, default=8)
    parser.add_argument("--progress-backtrack-tolerance", type=float, default=0.03)
    return parser.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def _problem_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row["dataset"]), str(row["problem_id"])


def _state_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    state = row.get("state_features", row)
    return state if isinstance(state, dict) else row


def _generated_tokens(row: Mapping[str, Any]) -> int:
    state = _state_payload(row)
    return int(state.get("generated_tokens", row.get("generated_tokens", row.get("token_count", 0))) or 0)


def _progress_ratio(row: Mapping[str, Any]) -> float:
    state = _state_payload(row)
    return float(state.get("progress_ratio", row.get("progress_ratio", 0.0)) or 0.0)


def _is_answer_zone(row: Mapping[str, Any]) -> bool:
    state = _state_payload(row)
    return bool(state.get("is_answer_zone", row.get("is_answer_zone", False)))


def _remaining_budget_tokens(row: Mapping[str, Any], cot_budget: int) -> int:
    state = _state_payload(row)
    ratio = float(state.get("remaining_budget_ratio", row.get("remaining_budget_ratio", 0.0)) or 0.0)
    return max(0, int(round(float(cot_budget) * max(0.0, min(1.0, ratio)))))


def _split_problem_keys(
    problem_keys: Sequence[Tuple[str, str]],
    *,
    seed: int,
    val_frac: float,
    disable_val: bool,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    ordered = sorted(set(problem_keys))
    if disable_val or len(ordered) <= 1 or float(val_frac) <= 0.0:
        return set(ordered), set()
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    val_count = max(1, int(round(len(ordered) * float(val_frac))))
    val_count = min(val_count, len(ordered) - 1)
    val_keys = set(ordered[:val_count])
    train_keys = set(ordered[val_count:])
    return train_keys, val_keys


def _load_action_space(payload: Mapping[str, Any]) -> JointActionSpace:
    action_cfg = payload["action_space"]
    return JointActionSpace(
        info_values=tuple(int(x) for x in action_cfg["info_values"]),
        cot_values=tuple(int(x) for x in action_cfg["cot_values"]),
        cot_token_budgets={int(k): int(v) for k, v in action_cfg["cot_token_budgets"].items()},
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


def _torch_load_artifact(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def _compute_problem_token_stats(scored_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], ProblemTokenStats]:
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for row in scored_rows:
        buckets[_problem_key(row)].append(int(row.get("token_count", 0) or 0))
    stats: Dict[Tuple[str, str], ProblemTokenStats] = {}
    for key, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        mean = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size else 1.0
        stats[key] = ProblemTokenStats(mean=mean, std=(std if std > 1e-6 else 1.0))
    return stats


def _recompute_score(*, correctness: float, token_count: int, stats: ProblemTokenStats, lambda_token_penalty: float) -> float:
    token_z = (int(token_count) - float(stats.mean)) / float(stats.std)
    return float(correctness) - (float(lambda_token_penalty) * token_z)


def _build_problem_states(
    scored_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], ProblemStaticState]:
    traces_by_problem: Dict[Tuple[str, str], Dict[Tuple[int, int], TraceRecord]] = defaultdict(dict)
    for row in scored_rows:
        key = _problem_key(row)
        action = (int(row["info_mode"]), int(row["cot_mode"]))
        traces_by_problem[key][action] = TraceRecord(
            dataset=str(row["dataset"]),
            problem_id=str(row["problem_id"]),
            prompt=str(row.get("prompt", "")),
            info_mode=int(row["info_mode"]),
            cot_mode=int(row["cot_mode"]),
            score=float(row.get("score", row.get("correctness", 0.0))),
            correctness=float(row.get("correctness", row.get("is_correct", 0.0))),
            token_count=int(row.get("token_count", 0) or 0),
            think_tokens=int(row.get("think_zone_token_count", 0) or 0),
            answer_tokens=int(row.get("answer_zone_token_count", 0) or 0),
            raw=dict(row),
        )
    boundaries_by_problem: Dict[Tuple[str, str], Dict[Tuple[int, int], List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in boundary_rows:
        key = _problem_key(row)
        action = (int(row["chosen_info_mode"]), int(row["chosen_cot_mode"]))
        boundaries_by_problem[key][action].append(dict(row))
    states: Dict[Tuple[str, str], ProblemStaticState] = {}
    for key, traces in traces_by_problem.items():
        prompt = next(iter(traces.values())).prompt if traces else ""
        boundary_map = boundaries_by_problem.get(key, {})
        for action_rows in boundary_map.values():
            action_rows.sort(key=lambda item: int(item.get("boundary_index", 0)))
        states[key] = ProblemStaticState(
            prompt=prompt,
            traces_by_action=traces,
            boundaries_by_action=dict(boundary_map),
        )
    return states


def _best_global_static_action(problem_states: Mapping[Tuple[str, str], ProblemStaticState], action_space: JointActionSpace) -> Tuple[int, int]:
    action_scores: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for state in problem_states.values():
        for action in action_space.actions:
            record = state.traces_by_action.get(action)
            if record is not None:
                action_scores[action].append(float(record.score))
    ranked = sorted(
        ((action, float(np.mean(scores))) for action, scores in action_scores.items() if scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0][0]


def _match_target_row(
    *,
    rows: Sequence[Dict[str, Any]],
    anchor_row: Mapping[str, Any],
    neighborhood_spec: BoundaryNeighborhoodSpec,
    progress_backtrack_tolerance: float,
) -> int:
    if not rows:
        return -1
    anchor_progress = _progress_ratio(anchor_row)
    best_idx = 0
    best_score = -1.0
    best_progress_gap = float("inf")
    for idx, row in enumerate(rows):
        progress = _progress_ratio(row)
        if progress + float(progress_backtrack_tolerance) < anchor_progress:
            continue
        weight = neighborhood_spec.candidate_weight(anchor_row, row)
        progress_gap = abs(progress - anchor_progress)
        if (weight > best_score) or (abs(weight - best_score) < 1e-8 and progress_gap < best_progress_gap):
            best_idx = idx
            best_score = weight
            best_progress_gap = progress_gap
    if best_score > 0.0:
        return best_idx
    fallback_idx = 0
    fallback_gap = float("inf")
    for idx, row in enumerate(rows):
        progress = _progress_ratio(row)
        if progress + float(progress_backtrack_tolerance) < anchor_progress:
            continue
        gap = abs(progress - anchor_progress)
        if gap < fallback_gap:
            fallback_idx = idx
            fallback_gap = gap
    return fallback_idx


def _evaluate_static_record(
    record: TraceRecord,
    *,
    stats: ProblemTokenStats,
    lambda_penalty: float,
) -> Dict[str, Any]:
    score = _recompute_score(
        correctness=record.correctness,
        token_count=record.token_count,
        stats=stats,
        lambda_token_penalty=lambda_penalty,
    )
    return {
        "score": score,
        "correctness": float(record.correctness),
        "token_count": int(record.token_count),
        "switch_count": 0,
        "start_action": {"info_mode": int(record.info_mode), "cot_mode": int(record.cot_mode)},
        "final_action": {"info_mode": int(record.info_mode), "cot_mode": int(record.cot_mode)},
    }


def _replay_problem(
    *,
    policy: OnlineDecisionPolicy,
    problem_state: ProblemStaticState,
    action_space: JointActionSpace,
    neighborhood_spec: BoundaryNeighborhoodSpec,
    token_stats: ProblemTokenStats,
    lambda_penalty: float,
    start_action: Tuple[int, int],
    max_switches: int,
    progress_backtrack_tolerance: float,
) -> Dict[str, Any]:
    current_action = (int(start_action[0]), int(start_action[1]))
    current_rows = problem_state.boundaries_by_action.get(current_action, [])
    if not current_rows:
        record = problem_state.traces_by_action[current_action]
        return _evaluate_static_record(record, stats=token_stats, lambda_penalty=lambda_penalty)

    current_pos = 0
    current_row = current_rows[current_pos]
    switch_count = 0
    step_count = 0
    consumed_tokens = _generated_tokens(current_row)
    max_steps = max(8, 2 * max(len(rows) for rows in problem_state.boundaries_by_action.values() if rows))

    while step_count < max_steps:
        step_count += 1
        remaining_budget_tokens = _remaining_budget_tokens(
            current_row,
            action_space.cot_token_budgets.get(int(current_action[1]), 0),
        )
        decision = policy.choose_boundary_action(
            prompt=problem_state.prompt,
            boundary_row=current_row,
            current_action=current_action,
            remaining_thinking_budget_tokens=remaining_budget_tokens,
        )
        next_action = (
            int(decision["best_action"]["info_mode"]),
            int(decision["best_action"]["cot_mode"]),
        )
        if next_action == current_action:
            next_pos = current_pos + 1
            if next_pos >= len(current_rows):
                break
            current_pos = next_pos
            current_row = current_rows[current_pos]
            consumed_tokens = max(consumed_tokens, _generated_tokens(current_row))
            continue

        if switch_count >= int(max_switches):
            break

        target_rows = problem_state.boundaries_by_action.get(next_action, [])
        if not target_rows:
            next_pos = current_pos + 1
            if next_pos >= len(current_rows):
                break
            current_pos = next_pos
            current_row = current_rows[current_pos]
            consumed_tokens = max(consumed_tokens, _generated_tokens(current_row))
            continue

        matched_pos = _match_target_row(
            rows=target_rows,
            anchor_row=current_row,
            neighborhood_spec=neighborhood_spec,
            progress_backtrack_tolerance=progress_backtrack_tolerance,
        )
        current_action = next_action
        current_rows = target_rows
        current_pos = max(0, matched_pos)
        current_row = current_rows[current_pos]
        consumed_tokens = max(consumed_tokens, _generated_tokens(current_row))
        switch_count += 1

    final_record = problem_state.traces_by_action[current_action]
    remaining_tail_tokens = max(0, int(final_record.token_count) - _generated_tokens(current_row))
    estimated_tokens = int(consumed_tokens + remaining_tail_tokens)
    estimated_score = _recompute_score(
        correctness=final_record.correctness,
        token_count=estimated_tokens,
        stats=token_stats,
        lambda_token_penalty=lambda_penalty,
    )
    return {
        "score": estimated_score,
        "correctness": float(final_record.correctness),
        "token_count": int(estimated_tokens),
        "switch_count": int(switch_count),
        "start_action": {"info_mode": int(start_action[0]), "cot_mode": int(start_action[1])},
        "final_action": {"info_mode": int(current_action[0]), "cot_mode": int(current_action[1])},
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"n": 0, "acc": 0.0, "score": 0.0, "tokens": 0.0, "switches": 0.0}
    return {
        "n": len(rows),
        "acc": float(np.mean([float(row["correctness"]) for row in rows])),
        "score": float(np.mean([float(row["score"]) for row in rows])),
        "tokens": float(np.mean([float(row["token_count"]) for row in rows])),
        "switches": float(np.mean([float(row.get("switch_count", 0.0)) for row in rows])),
    }


def main() -> int:
    args = parse_args()
    supervision_dir = Path(args.supervision_dir).expanduser().resolve()
    artifact_path = Path(args.artifact).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("Loading training artifact...")
    artifact = _torch_load_artifact(artifact_path)
    action_space = _load_action_space(artifact)
    text_vectorizer = HashTextVectorizer(**artifact["text_vectorizer"])
    slot_memory = _load_slot_memory(artifact)
    boundary_spec = _load_boundary_spec(artifact)
    neighborhood_spec = _load_neighborhood_spec(artifact)

    prior_model = _load_model(artifact, "prior_model")
    think_boundary_model = _load_model(artifact, "think_boundary_model")
    answer_boundary_model = _load_model(artifact, "answer_boundary_model")
    policy = OnlineDecisionPolicy(
        action_space=action_space,
        text_vectorizer=text_vectorizer,
        boundary_spec=boundary_spec,
        slot_memory=slot_memory,
        prior_model=prior_model,
        think_boundary_model=think_boundary_model,
        answer_boundary_model=answer_boundary_model,
        prior_weight=float(args.prior_weight),
        boundary_weight=float(args.boundary_weight),
        switch_cost=float(args.switch_cost),
        hysteresis_bonus=float(args.hysteresis_bonus),
        budget_guardrail_penalty=float(args.budget_guardrail_penalty),
    )

    _log("Loading offline supervision rows...")
    scored_rows = read_jsonl(supervision_dir / "offline_supervision" / "scored_traces.jsonl")
    boundary_rows = read_jsonl(supervision_dir / "offline_supervision" / "boundary_samples.jsonl")
    problem_states = _build_problem_states(scored_rows, boundary_rows)
    token_stats = _compute_problem_token_stats(scored_rows)
    lambda_penalty = _infer_lambda(scored_rows)

    train_cfg = artifact.get("train_config", {})
    split_seed = int(args.split_seed if args.split_seed is not None else artifact.get("seed", 20260412))
    val_problem_frac = float(
        args.val_problem_frac
        if args.val_problem_frac is not None
        else train_cfg.get("val_problem_frac", 0.15)
    )
    disable_val = bool(args.disable_val or train_cfg.get("disable_val", False))
    train_keys, val_keys = _split_problem_keys(
        list(problem_states.keys()),
        seed=split_seed,
        val_frac=val_problem_frac,
        disable_val=disable_val,
    )
    if args.split == "train":
        selected_keys = train_keys
    elif args.split == "val":
        selected_keys = val_keys
    else:
        selected_keys = set(problem_states.keys())

    selected_states = {key: state for key, state in problem_states.items() if key in selected_keys}
    _log(f"Evaluating split={args.split} with {len(selected_states)} problems...")

    best_static_action = _best_global_static_action(selected_states, action_space)

    rows_replay: List[Dict[str, Any]] = []
    rows_prior_only: List[Dict[str, Any]] = []
    rows_best_static: List[Dict[str, Any]] = []
    rows_oracle: List[Dict[str, Any]] = []

    for key in sorted(selected_states):
        state = selected_states[key]
        stats = token_stats[key]
        prompt = state.prompt

        prior_scores = policy.score_initial_actions(prompt)
        prior_idx = int(np.argmax(prior_scores))
        prior_action = action_space.index_to_action(prior_idx)
        prior_record = state.traces_by_action[prior_action]
        prior_eval = _evaluate_static_record(prior_record, stats=stats, lambda_penalty=lambda_penalty)
        prior_eval.update({"dataset": key[0], "problem_id": key[1], "policy": "prior_only"})
        rows_prior_only.append(prior_eval)

        replay_eval = _replay_problem(
            policy=policy,
            problem_state=state,
            action_space=action_space,
            neighborhood_spec=neighborhood_spec,
            token_stats=stats,
            lambda_penalty=lambda_penalty,
            start_action=prior_action,
            max_switches=int(args.max_switches),
            progress_backtrack_tolerance=float(args.progress_backtrack_tolerance),
        )
        replay_eval.update({"dataset": key[0], "problem_id": key[1], "policy": "replay"})
        rows_replay.append(replay_eval)

        best_static_record = state.traces_by_action[best_static_action]
        best_static_eval = _evaluate_static_record(best_static_record, stats=stats, lambda_penalty=lambda_penalty)
        best_static_eval.update({"dataset": key[0], "problem_id": key[1], "policy": "best_static"})
        rows_best_static.append(best_static_eval)

        oracle_record = max(state.traces_by_action.values(), key=lambda item: float(item.score))
        oracle_eval = _evaluate_static_record(oracle_record, stats=stats, lambda_penalty=lambda_penalty)
        oracle_eval.update({"dataset": key[0], "problem_id": key[1], "policy": "oracle"})
        rows_oracle.append(oracle_eval)

    by_name = {
        "replay": rows_replay,
        "prior_only": rows_prior_only,
        "best_static": rows_best_static,
        "oracle": rows_oracle,
    }

    oracle_by_problem = {(row["dataset"], row["problem_id"]): row for row in rows_oracle}
    best_static_by_problem = {(row["dataset"], row["problem_id"]): row for row in rows_best_static}
    prior_only_by_problem = {(row["dataset"], row["problem_id"]): row for row in rows_prior_only}

    summary = {
        "split": args.split,
        "problem_count": len(selected_states),
        "lambda_token_penalty": float(lambda_penalty),
        "best_static_action": {"info_mode": int(best_static_action[0]), "cot_mode": int(best_static_action[1])},
        "metrics": {name: _summarize_rows(rows) for name, rows in by_name.items()},
        "comparisons": {
            "replay_vs_best_static": {
                "score_delta": float(
                    np.mean([rows_replay[i]["score"] - rows_best_static[i]["score"] for i in range(len(rows_replay))])
                )
                if rows_replay
                else 0.0,
                "win_rate": float(
                    np.mean([rows_replay[i]["score"] > rows_best_static[i]["score"] for i in range(len(rows_replay))])
                )
                if rows_replay
                else 0.0,
            },
            "replay_vs_prior_only": {
                "score_delta": float(
                    np.mean([rows_replay[i]["score"] - rows_prior_only[i]["score"] for i in range(len(rows_replay))])
                )
                if rows_replay
                else 0.0,
                "win_rate": float(
                    np.mean([rows_replay[i]["score"] > rows_prior_only[i]["score"] for i in range(len(rows_replay))])
                )
                if rows_replay
                else 0.0,
            },
            "oracle_regret": float(
                np.mean(
                    [
                        oracle_by_problem[(row["dataset"], row["problem_id"])]["score"] - row["score"]
                        for row in rows_replay
                    ]
                )
            )
            if rows_replay
            else 0.0,
        },
    }

    per_problem_rows: List[Dict[str, Any]] = []
    for replay_row in rows_replay:
        key = (replay_row["dataset"], replay_row["problem_id"])
        per_problem_rows.append(
            {
                "dataset": key[0],
                "problem_id": key[1],
                "replay": replay_row,
                "prior_only": prior_only_by_problem[key],
                "best_static": best_static_by_problem[key],
                "oracle": oracle_by_problem[key],
            }
        )

    write_json(output_dir / "replay_summary.json", summary)
    write_jsonl(output_dir / "replay_per_problem.jsonl", per_problem_rows)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
