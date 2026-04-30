from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .action_space import JointActionSpace
from .neighborhood import (
    DEFAULT_NEIGHBORHOOD_CONFIG,
    BoundaryNeighborhoodConfig,
    BoundaryNeighborhoodSpec,
)


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-6, float(temperature))
    z = scores / t
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.maximum(1e-8, exp_z.sum())


def _problem_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row["dataset"]), str(row["problem_id"])


def _state_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    state = row.get("state_features", row)
    return state if isinstance(state, dict) else row


def _is_answer_zone(row: Mapping[str, Any]) -> bool:
    state = _state_payload(row)
    return bool(state.get("is_answer_zone", row.get("is_answer_zone", False)))


def _trajectory_score(row: Mapping[str, Any]) -> float:
    return float(row.get("score", row.get("trajectory_score", row.get("correctness", 0.0))) or 0.0)


@dataclass(frozen=True)
class PriorExample:
    dataset: str
    problem_id: str
    prompt: str
    action_scores: np.ndarray
    action_distribution: np.ndarray
    weight: float


@dataclass(frozen=True)
class BoundaryExample:
    dataset: str
    problem_id: str
    boundary_index: int
    prompt: str
    phase: str
    current_info_mode: int
    current_cot_mode: int
    target_distribution: np.ndarray
    weight: float
    row: Dict[str, Any]


@dataclass(frozen=True)
class BoundaryPair:
    example_index: int
    winner_index: int
    loser_index: int
    weight: float


@dataclass(frozen=True)
class BoundaryTrainingBundle:
    think_examples: List[BoundaryExample]
    think_pairs: List[BoundaryPair]
    answer_examples: List[BoundaryExample]
    answer_pairs: List[BoundaryPair]
    neighborhood_spec: BoundaryNeighborhoodSpec


@dataclass(frozen=True)
class _ProblemBoundaryResult:
    think_examples: List[BoundaryExample]
    think_pairs: List[BoundaryPair]
    answer_examples: List[BoundaryExample]
    answer_pairs: List[BoundaryPair]
    row_count: int


def build_prior_examples(
    scored_traces: Sequence[Dict[str, Any]],
    action_space: JointActionSpace,
    *,
    target_temperature: float = 0.25,
    missing_penalty: float = 2.0,
) -> List[PriorExample]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_traces:
        grouped[_problem_key(row)].append(row)

    examples: List[PriorExample] = []
    for (dataset, problem_id), rows in grouped.items():
        action_scores = np.full((action_space.size,), -missing_penalty, dtype=np.float32)
        prompt = str(rows[0].get("prompt", ""))
        seen = np.zeros((action_space.size,), dtype=np.float32)
        for row in rows:
            idx = action_space.action_to_index(int(row["info_mode"]), int(row["cot_mode"]))
            action_scores[idx] = float(row.get("score", row.get("correctness", 0.0)))
            seen[idx] = 1.0
        min_seen = float(action_scores[seen > 0].min()) if float(seen.sum()) > 0 else 0.0
        action_scores[seen <= 0] = min_seen - float(missing_penalty)
        distribution = _softmax(action_scores, temperature=target_temperature).astype(np.float32)
        weight = float(action_scores.max() - action_scores.min())
        examples.append(
            PriorExample(
                dataset=dataset,
                problem_id=problem_id,
                prompt=prompt,
                action_scores=action_scores,
                action_distribution=distribution,
                weight=max(0.1, weight),
            )
        )
    return examples


def _local_joint_scores(
    *,
    anchor_row: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    action_space: JointActionSpace,
    neighborhood_spec: BoundaryNeighborhoodSpec,
    missing_penalty: float,
) -> np.ndarray:
    weighted_neighbors: List[Tuple[float, Mapping[str, Any]]] = []
    for candidate in candidate_rows:
        weight = neighborhood_spec.candidate_weight(anchor_row, candidate)
        if weight > 0.0:
            weighted_neighbors.append((weight, candidate))
    weighted_neighbors.sort(key=lambda item: item[0], reverse=True)
    weighted_neighbors = weighted_neighbors[: max(1, int(neighborhood_spec.config.top_k))]

    score_sums = np.zeros((action_space.size,), dtype=np.float32)
    weight_sums = np.zeros((action_space.size,), dtype=np.float32)
    for weight, candidate in weighted_neighbors:
        idx = action_space.action_to_index(
            int(candidate["chosen_info_mode"]),
            int(candidate["chosen_cot_mode"]),
        )
        score_sums[idx] += float(weight) * _trajectory_score(candidate)
        weight_sums[idx] += float(weight)

    action_scores = np.zeros((action_space.size,), dtype=np.float32)
    seen = weight_sums > 0
    if np.any(seen):
        action_scores[seen] = score_sums[seen] / np.maximum(1e-8, weight_sums[seen])
        min_seen = float(action_scores[seen].min())
    else:
        min_seen = 0.0
    action_scores[~seen] = min_seen - float(missing_penalty)
    return action_scores


def _candidate_rows_for_anchor(
    *,
    anchor_row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    buckets: Mapping[Tuple[str, int], Sequence[Mapping[str, Any]]],
    neighborhood_spec: BoundaryNeighborhoodSpec,
) -> Sequence[Mapping[str, Any]]:
    bucket_ids = neighborhood_spec.neighboring_bucket_ids(anchor_row)
    candidates: List[Mapping[str, Any]] = []
    seen_ids = set()
    for bucket_id in bucket_ids:
        for candidate in buckets.get(bucket_id, ()):
            marker = id(candidate)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            candidates.append(candidate)
    return candidates if candidates else rows


def _collapse_joint_scores_to_info(
    joint_scores: np.ndarray,
    action_space: JointActionSpace,
) -> np.ndarray:
    info_scores = np.zeros((action_space.info_size,), dtype=np.float32)
    for info_idx, info_mode in enumerate(action_space.info_values):
        joint_indices = action_space.joint_indices_for_info(int(info_mode))
        info_scores[info_idx] = float(np.mean(joint_scores[joint_indices]))
    return info_scores


def _build_pairs_from_scores(
    *,
    example_index: int,
    scores: np.ndarray,
    min_score_gap: float,
    max_pairs: int,
) -> List[BoundaryPair]:
    ranked = sorted(
        [(idx, float(score)) for idx, score in enumerate(scores)],
        key=lambda item: item[1],
        reverse=True,
    )
    local_pairs: List[BoundaryPair] = []
    if not ranked:
        return local_pairs
    spread = max(1e-6, ranked[0][1] - ranked[-1][1])
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            winner_index, winner_score = ranked[i]
            loser_index, loser_score = ranked[j]
            gap = winner_score - loser_score
            if gap < float(min_score_gap):
                continue
            local_pairs.append(
                BoundaryPair(
                    example_index=example_index,
                    winner_index=int(winner_index),
                    loser_index=int(loser_index),
                    weight=float(gap / spread),
                )
            )
    local_pairs.sort(key=lambda item: float(item.weight), reverse=True)
    return local_pairs[: max(1, int(max_pairs))]


def build_problem_boundary_examples_for_rows(
    *,
    rows: Sequence[Dict[str, Any]],
    action_space: JointActionSpace,
    neighborhood_spec: BoundaryNeighborhoodSpec,
    target_temperature: float,
    missing_penalty: float,
    pair_min_score_gap: float,
    max_pairs_per_example: int,
) -> _ProblemBoundaryResult:
    think_examples: List[BoundaryExample] = []
    think_pairs: List[BoundaryPair] = []
    answer_examples: List[BoundaryExample] = []
    answer_pairs: List[BoundaryPair] = []
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for candidate in rows:
        buckets[neighborhood_spec.coarse_bucket_id(candidate)].append(candidate)

    for row in rows:
        candidate_rows = _candidate_rows_for_anchor(
            anchor_row=row,
            rows=rows,
            buckets=buckets,
            neighborhood_spec=neighborhood_spec,
        )
        joint_scores = _local_joint_scores(
            anchor_row=row,
            candidate_rows=candidate_rows,
            action_space=action_space,
            neighborhood_spec=neighborhood_spec,
            missing_penalty=missing_penalty,
        )
        if _is_answer_zone(row):
            target_scores = _collapse_joint_scores_to_info(joint_scores, action_space)
            distribution = _softmax(target_scores, temperature=target_temperature).astype(np.float32)
            example_index = len(answer_examples)
            answer_examples.append(
                BoundaryExample(
                    dataset=str(row["dataset"]),
                    problem_id=str(row["problem_id"]),
                    boundary_index=int(row["boundary_index"]),
                    prompt=str(row.get("prompt", "")),
                    phase="answer",
                    current_info_mode=int(row["chosen_info_mode"]),
                    current_cot_mode=int(row["chosen_cot_mode"]),
                    target_distribution=distribution,
                    weight=max(0.1, float(target_scores.max() - target_scores.min())),
                    row=row,
                )
            )
            answer_pairs.extend(
                _build_pairs_from_scores(
                    example_index=example_index,
                    scores=target_scores,
                    min_score_gap=pair_min_score_gap,
                    max_pairs=max_pairs_per_example,
                )
            )
        else:
            distribution = _softmax(joint_scores, temperature=target_temperature).astype(np.float32)
            example_index = len(think_examples)
            think_examples.append(
                BoundaryExample(
                    dataset=str(row["dataset"]),
                    problem_id=str(row["problem_id"]),
                    boundary_index=int(row["boundary_index"]),
                    prompt=str(row.get("prompt", "")),
                    phase="think",
                    current_info_mode=int(row["chosen_info_mode"]),
                    current_cot_mode=int(row["chosen_cot_mode"]),
                    target_distribution=distribution,
                    weight=max(0.1, float(joint_scores.max() - joint_scores.min())),
                    row=row,
                )
            )
            think_pairs.extend(
                _build_pairs_from_scores(
                    example_index=example_index,
                    scores=joint_scores,
                    min_score_gap=pair_min_score_gap,
                    max_pairs=max_pairs_per_example,
                )
            )
    return _ProblemBoundaryResult(
        think_examples=think_examples,
        think_pairs=think_pairs,
        answer_examples=answer_examples,
        answer_pairs=answer_pairs,
        row_count=len(rows),
    )


def _reindex_pairs(
    pairs: Sequence[BoundaryPair],
    *,
    index_offset: int,
) -> List[BoundaryPair]:
    return [
        BoundaryPair(
            example_index=int(pair.example_index) + int(index_offset),
            winner_index=int(pair.winner_index),
            loser_index=int(pair.loser_index),
            weight=float(pair.weight),
        )
        for pair in pairs
    ]


def build_boundary_examples(
    boundary_samples: Sequence[Dict[str, Any]],
    boundary_preference_pairs: Sequence[Dict[str, Any]],
    action_space: JointActionSpace,
    *,
    target_temperature: float = 0.25,
    missing_penalty: float = 2.0,
    neighborhood_config: BoundaryNeighborhoodConfig = DEFAULT_NEIGHBORHOOD_CONFIG,
    pair_min_score_gap: float = 0.12,
    max_pairs_per_example: int = 6,
    progress_callback: Callable[[int, int], None] | None = None,
    num_workers: int = 1,
    worker_start_method: str = "spawn",
    max_tasks_in_flight: int | None = None,
) -> BoundaryTrainingBundle:
    del boundary_preference_pairs

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in boundary_samples:
        grouped[_problem_key(row)].append(row)

    neighborhood_spec = BoundaryNeighborhoodSpec.fit(boundary_samples, neighborhood_config)
    think_examples: List[BoundaryExample] = []
    think_pairs: List[BoundaryPair] = []
    answer_examples: List[BoundaryExample] = []
    answer_pairs: List[BoundaryPair] = []
    total_rows = sum(len(rows) for rows in grouped.values())
    processed_rows = 0
    grouped_items = sorted(grouped.items(), key=lambda item: item[0])

    def _merge_problem_result(result: _ProblemBoundaryResult) -> None:
        nonlocal processed_rows
        think_offset = len(think_examples)
        answer_offset = len(answer_examples)
        think_examples.extend(result.think_examples)
        think_pairs.extend(_reindex_pairs(result.think_pairs, index_offset=think_offset))
        answer_examples.extend(result.answer_examples)
        answer_pairs.extend(_reindex_pairs(result.answer_pairs, index_offset=answer_offset))
        processed_rows += int(result.row_count)
        if progress_callback is not None:
            progress_callback(processed_rows, total_rows)

    if int(num_workers) <= 1:
        for _, rows in grouped_items:
            result = build_problem_boundary_examples_for_rows(
                rows=rows,
                action_space=action_space,
                neighborhood_spec=neighborhood_spec,
                target_temperature=target_temperature,
                missing_penalty=missing_penalty,
                pair_min_score_gap=pair_min_score_gap,
                max_pairs_per_example=max_pairs_per_example,
            )
            _merge_problem_result(result)
    else:
        inflight_limit = int(max_tasks_in_flight or max(1, int(num_workers) * 2))
        ctx = get_context(str(worker_start_method))
        with ProcessPoolExecutor(max_workers=int(num_workers), mp_context=ctx) as executor:
            iterator = iter(grouped_items)
            in_flight = set()

            def _submit_next() -> bool:
                try:
                    _, rows = next(iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    build_problem_boundary_examples_for_rows,
                    rows=rows,
                    action_space=action_space,
                    neighborhood_spec=neighborhood_spec,
                    target_temperature=target_temperature,
                    missing_penalty=missing_penalty,
                    pair_min_score_gap=pair_min_score_gap,
                    max_pairs_per_example=max_pairs_per_example,
                )
                in_flight.add(future)
                return True

            while len(in_flight) < inflight_limit and _submit_next():
                pass

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    _merge_problem_result(future.result())
                while len(in_flight) < inflight_limit and _submit_next():
                    pass

    return BoundaryTrainingBundle(
        think_examples=think_examples,
        think_pairs=think_pairs,
        answer_examples=answer_examples,
        answer_pairs=answer_pairs,
        neighborhood_spec=neighborhood_spec,
    )
