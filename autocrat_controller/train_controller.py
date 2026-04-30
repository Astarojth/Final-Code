#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT.parent))
    from autocrat_controller.action_space import DEFAULT_ACTION_SPACE
    from autocrat_controller.datasets import (
        BoundaryExample,
        BoundaryPair,
        PriorExample,
        build_prior_examples,
        build_problem_boundary_examples_for_rows,
    )
    from autocrat_controller.features import BoundaryFeatureSpec, HashTextVectorizer
    from autocrat_controller.io_utils import read_jsonl, write_json
    from autocrat_controller.models import MLP, pairwise_logistic_loss
    from autocrat_controller.neighborhood import DEFAULT_NEIGHBORHOOD_CONFIG, BoundaryNeighborhoodConfig, BoundaryNeighborhoodSpec
    from autocrat_controller.slot_memory import SlotMemory
else:
    from .action_space import DEFAULT_ACTION_SPACE
    from .datasets import (
        BoundaryExample,
        BoundaryPair,
        PriorExample,
        build_prior_examples,
        build_problem_boundary_examples_for_rows,
    )
    from .features import BoundaryFeatureSpec, HashTextVectorizer
    from .io_utils import read_jsonl, write_json
    from .models import MLP, pairwise_logistic_loss
    from .neighborhood import DEFAULT_NEIGHBORHOOD_CONFIG, BoundaryNeighborhoodConfig, BoundaryNeighborhoodSpec
    from .slot_memory import SlotMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AutoCRAT boundary controller from offline supervision artifacts.")
    parser.add_argument("--supervision-dir", required=True, help="Path to the offline supervision root.")
    parser.add_argument("--output-dir", required=True, help="Output directory for new independent training artifacts.")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--slot-temperature", type=float, default=0.35)
    parser.add_argument("--target-temperature", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prior-epochs", type=int, default=40)
    parser.add_argument("--think-boundary-epochs", type=int, default=100)
    parser.add_argument("--answer-boundary-epochs", type=int, default=100)
    parser.add_argument("--prior-log-every", type=int, default=10)
    parser.add_argument("--boundary-log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--neighborhood-top-k", type=int, default=int(DEFAULT_NEIGHBORHOOD_CONFIG.top_k))
    parser.add_argument("--neighborhood-temperature", type=float, default=float(DEFAULT_NEIGHBORHOOD_CONFIG.kernel_temperature))
    parser.add_argument(
        "--neighborhood-bucket-strategy",
        choices=("none", "progress_ratio", "boundary_index"),
        default=str(DEFAULT_NEIGHBORHOOD_CONFIG.coarse_bucket_strategy),
    )
    parser.add_argument("--progress-bucket-count", type=int, default=int(DEFAULT_NEIGHBORHOOD_CONFIG.progress_bucket_count))
    parser.add_argument("--progress-bucket-radius", type=int, default=int(DEFAULT_NEIGHBORHOOD_CONFIG.progress_bucket_radius))
    parser.add_argument("--boundary-index-bucket-size", type=int, default=int(DEFAULT_NEIGHBORHOOD_CONFIG.boundary_index_bucket_size))
    parser.add_argument("--boundary-index-bucket-radius", type=int, default=int(DEFAULT_NEIGHBORHOOD_CONFIG.boundary_index_bucket_radius))
    parser.add_argument("--neighborhood-pair-gap", type=float, default=0.12)
    parser.add_argument("--neighborhood-max-pairs", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-start-method", choices=("spawn", "fork", "forkserver"), default="spawn")
    parser.add_argument("--max-tasks-in-flight", type=int, default=0)
    parser.add_argument("--boundary-shard-buffer-mb", type=int, default=0)
    parser.add_argument("--boundary-buffer-fraction", type=float, default=0.08)
    parser.add_argument("--boundary-buffer-max-mb", type=int, default=768)
    parser.add_argument("--keep-boundary-cache", action="store_true")
    parser.add_argument("--val-problem-frac", type=float, default=0.15)
    parser.add_argument("--disable-val", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(x: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype)


def _empty_array(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.float32)


def _log(message: str) -> None:
    print(message, flush=True)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def _problem_key_from_example(row) -> Tuple[str, str]:
    return str(row.dataset), str(row.problem_id)


def _problem_key_from_raw(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row["dataset"]), str(row["problem_id"])


def _split_problem_keys(
    prior_examples: Sequence[PriorExample],
    *,
    seed: int,
    val_frac: float,
    disable_val: bool,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    all_keys = sorted({_problem_key_from_example(row) for row in prior_examples})
    if disable_val or len(all_keys) <= 1 or float(val_frac) <= 0.0:
        return set(all_keys), set()
    rng = random.Random(int(seed))
    rng.shuffle(all_keys)
    val_count = max(1, int(round(len(all_keys) * float(val_frac))))
    val_count = min(val_count, len(all_keys) - 1)
    val_keys = set(all_keys[:val_count])
    train_keys = set(all_keys[val_count:])
    return train_keys, val_keys


def _filter_prior_examples(examples: Sequence[PriorExample], allowed_keys: Set[Tuple[str, str]]) -> List[PriorExample]:
    return [row for row in examples if _problem_key_from_example(row) in allowed_keys]


def _prepare_prior_inputs(examples: Sequence[PriorExample], text_vectorizer, slot_memory):
    if not examples:
        input_dim = int(text_vectorizer.dim) + DEFAULT_ACTION_SPACE.size
        return _empty_array(0, input_dim), _empty_array(0, DEFAULT_ACTION_SPACE.size), np.zeros((0,), dtype=np.float32)
    prompts = [row.prompt for row in examples]
    prompt_x = text_vectorizer.transform(prompts)
    slot_prior = slot_memory.query(prompt_x)
    x = np.concatenate([prompt_x, slot_prior], axis=1).astype(np.float32)
    y = np.stack([row.action_distribution for row in examples], axis=0).astype(np.float32)
    weights = np.asarray([row.weight for row in examples], dtype=np.float32)
    return x, y, weights


def _prepare_think_boundary_inputs(
    examples: Sequence[BoundaryExample],
    text_vectorizer: HashTextVectorizer,
    slot_memory: SlotMemory,
    boundary_spec: BoundaryFeatureSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        input_dim = int(text_vectorizer.dim) + DEFAULT_ACTION_SPACE.size + boundary_spec.dim + DEFAULT_ACTION_SPACE.size
        return _empty_array(0, input_dim), _empty_array(0, DEFAULT_ACTION_SPACE.size), np.zeros((0,), dtype=np.float32)
    prompts = [row.prompt for row in examples]
    prompt_x = text_vectorizer.transform(prompts)
    slot_prior = slot_memory.query(prompt_x)
    boundary_x = boundary_spec.transform([row.row for row in examples])
    current_x = np.asarray(
        [DEFAULT_ACTION_SPACE.current_action_onehot(row.current_info_mode, row.current_cot_mode) for row in examples],
        dtype=np.float32,
    )
    x = np.concatenate([prompt_x, slot_prior, boundary_x, current_x], axis=1).astype(np.float32)
    y = np.stack([row.target_distribution for row in examples], axis=0).astype(np.float32)
    weights = np.asarray([row.weight for row in examples], dtype=np.float32)
    return x, y, weights


def _prepare_answer_boundary_inputs(
    examples: Sequence[BoundaryExample],
    text_vectorizer: HashTextVectorizer,
    slot_memory: SlotMemory,
    boundary_spec: BoundaryFeatureSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        input_dim = int(text_vectorizer.dim) + DEFAULT_ACTION_SPACE.size + boundary_spec.dim + DEFAULT_ACTION_SPACE.info_size
        return _empty_array(0, input_dim), _empty_array(0, DEFAULT_ACTION_SPACE.info_size), np.zeros((0,), dtype=np.float32)
    prompts = [row.prompt for row in examples]
    prompt_x = text_vectorizer.transform(prompts)
    slot_prior = slot_memory.query(prompt_x)
    boundary_x = boundary_spec.transform([row.row for row in examples])
    current_x = np.asarray([DEFAULT_ACTION_SPACE.current_info_onehot(row.current_info_mode) for row in examples], dtype=np.float32)
    x = np.concatenate([prompt_x, slot_prior, boundary_x, current_x], axis=1).astype(np.float32)
    y = np.stack([row.target_distribution for row in examples], axis=0).astype(np.float32)
    weights = np.asarray([row.weight for row in examples], dtype=np.float32)
    return x, y, weights


def evaluate_prior_model(model: MLP, *, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    if x.shape[0] == 0:
        return {"loss": 0.0, "top1_match": 0.0}
    x_t = _to_tensor(x)
    y_t = _to_tensor(y)
    w_t = _to_tensor(weights / np.maximum(1e-6, weights.mean()))
    with torch.no_grad():
        logits = model(x_t)
        preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(y_t, dim=-1)
        acc = float((preds == labels).float().mean().item())
        loss = float((-(y_t * torch.log_softmax(logits, dim=-1)).sum(dim=-1) * w_t).mean().item())
    return {"loss": loss, "top1_match": acc}


def train_prior_model(
    model: MLP,
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    log_every: int = 0,
    log_prefix: str = "prior",
) -> Dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = _to_tensor(x)
    y_t = _to_tensor(y)
    w_t = _to_tensor(weights / np.maximum(1e-6, weights.mean()))
    start = time.perf_counter()
    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        logits = model(x_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        losses = -(y_t * log_probs).sum(dim=-1)
        loss = (losses * w_t).mean()
        loss.backward()
        optimizer.step()
        if int(log_every) > 0 and (((epoch + 1) % int(log_every) == 0) or (epoch + 1 == int(epochs))):
            _log(f"  [{log_prefix}] epoch {epoch + 1}/{int(epochs)} loss={float(loss.item()):.4f} elapsed={_format_seconds(time.perf_counter() - start)}")
    return evaluate_prior_model(model, x=x, y=y, weights=weights)


def _estimate_available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    return 4 * 1024**3


def _determine_boundary_buffer_bytes(args: argparse.Namespace) -> int:
    if int(args.boundary_shard_buffer_mb) > 0:
        return int(args.boundary_shard_buffer_mb) * 1024 * 1024
    available = _estimate_available_memory_bytes()
    frac = max(0.01, float(args.boundary_buffer_fraction))
    cap = max(128, int(args.boundary_buffer_max_mb)) * 1024 * 1024
    target = int(available * frac)
    target = max(128 * 1024 * 1024, min(cap, target))
    return target


def _override_neighborhood_config(args: argparse.Namespace) -> BoundaryNeighborhoodConfig:
    features = tuple(DEFAULT_NEIGHBORHOOD_CONFIG.features)
    return BoundaryNeighborhoodConfig(
        features=features,
        top_k=int(args.neighborhood_top_k),
        kernel_temperature=float(args.neighborhood_temperature),
        coarse_bucket_strategy=str(args.neighborhood_bucket_strategy),
        progress_bucket_count=int(args.progress_bucket_count),
        progress_bucket_radius=int(args.progress_bucket_radius),
        boundary_index_bucket_size=int(args.boundary_index_bucket_size),
        boundary_index_bucket_radius=int(args.boundary_index_bucket_radius),
        require_same_answer_zone=bool(DEFAULT_NEIGHBORHOOD_CONFIG.require_same_answer_zone),
        require_same_boundary_kind=bool(DEFAULT_NEIGHBORHOOD_CONFIG.require_same_boundary_kind),
        min_kernel_weight=float(DEFAULT_NEIGHBORHOOD_CONFIG.min_kernel_weight),
    )


def _pair_arrays(rows: Sequence[BoundaryPair]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row.example_index for row in rows], dtype=np.int64),
        np.asarray([row.winner_index for row in rows], dtype=np.int64),
        np.asarray([row.loser_index for row in rows], dtype=np.int64),
        np.asarray([row.weight for row in rows], dtype=np.float32),
    )


def _iter_problem_results(
    grouped_items: Sequence[Tuple[Tuple[str, str], List[Dict[str, Any]]]],
    *,
    action_space,
    neighborhood_spec,
    target_temperature: float,
    missing_penalty: float,
    pair_min_score_gap: float,
    max_pairs_per_example: int,
    num_workers: int,
    worker_start_method: str,
    max_tasks_in_flight: int | None,
    progress_callback,
):
    total_rows = sum(len(rows) for _, rows in grouped_items)
    processed_rows = 0

    def _emit(result):
        nonlocal processed_rows
        processed_rows += int(result.row_count)
        if progress_callback is not None:
            progress_callback(processed_rows, total_rows)
        return result

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
            yield _emit(result)
        return

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
                yield _emit(future.result())
            while len(in_flight) < inflight_limit and _submit_next():
                pass


def _append_pairs_with_offset(
    pair_arrays,
    pairs: Sequence[BoundaryPair],
    *,
    example_offset: int,
) -> int:
    if not pairs:
        return 0
    ex_idx, winner_idx, loser_idx, pair_w = _pair_arrays(pairs)
    pair_arrays["example_index"].append(ex_idx + int(example_offset))
    pair_arrays["winner_index"].append(winner_idx)
    pair_arrays["loser_index"].append(loser_idx)
    pair_arrays["weight"].append(pair_w)
    return int(ex_idx.shape[0])


def _save_shard(path: Path, *, x, y, weights, pair_arrays) -> Dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pair_arrays["example_index"]:
        example_index = np.concatenate(pair_arrays["example_index"], axis=0).astype(np.int64)
        winner_index = np.concatenate(pair_arrays["winner_index"], axis=0).astype(np.int64)
        loser_index = np.concatenate(pair_arrays["loser_index"], axis=0).astype(np.int64)
        pair_weight = np.concatenate(pair_arrays["weight"], axis=0).astype(np.float32)
    else:
        example_index = np.zeros((0,), dtype=np.int64)
        winner_index = np.zeros((0,), dtype=np.int64)
        loser_index = np.zeros((0,), dtype=np.int64)
        pair_weight = np.zeros((0,), dtype=np.float32)
    np.savez(
        path,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        weights=weights.astype(np.float32),
        pair_example_index=example_index,
        pair_winner_index=winner_index,
        pair_loser_index=loser_index,
        pair_weight=pair_weight,
    )
    return {"examples": int(x.shape[0]), "pairs": int(example_index.shape[0])}


def _empty_pair_buffers():
    return {
        "example_index": [],
        "winner_index": [],
        "loser_index": [],
        "weight": [],
    }


def _materialize_boundary_shards(
    *,
    boundary_samples: Sequence[Dict[str, Any]],
    train_keys: Set[Tuple[str, str]],
    val_keys: Set[Tuple[str, str]],
    action_space,
    neighborhood_config: BoundaryNeighborhoodConfig,
    target_temperature: float,
    pair_min_score_gap: float,
    max_pairs_per_example: int,
    text_vectorizer: HashTextVectorizer,
    slot_memory: SlotMemory,
    boundary_spec: BoundaryFeatureSpec,
    cache_dir: Path,
    progress_callback,
    num_workers: int,
    worker_start_method: str,
    max_tasks_in_flight: int | None,
    buffer_bytes_limit: int,
) -> Tuple[Dict[str, Any], BoundaryNeighborhoodSpec]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in boundary_samples:
        grouped.setdefault(_problem_key_from_raw(row), []).append(row)
    grouped_items = sorted(grouped.items(), key=lambda item: item[0])
    neighborhood_spec = BoundaryNeighborhoodSpec.fit(boundary_samples, neighborhood_config)

    manifests: Dict[str, Dict[str, Any]] = {
        "train_think": {"paths": [], "examples": 0, "pairs": 0, "shards": 0},
        "val_think": {"paths": [], "examples": 0, "pairs": 0, "shards": 0},
        "train_answer": {"paths": [], "examples": 0, "pairs": 0, "shards": 0},
        "val_answer": {"paths": [], "examples": 0, "pairs": 0, "shards": 0},
    }

    buffers = {
        key: {
            "x": [],
            "y": [],
            "weights": [],
            "pairs": _empty_pair_buffers(),
            "examples": 0,
            "bytes": 0,
        }
        for key in manifests
    }

    def _flush(name: str) -> None:
        buf = buffers[name]
        if not buf["x"]:
            return
        x = np.concatenate(buf["x"], axis=0)
        y = np.concatenate(buf["y"], axis=0)
        weights = np.concatenate(buf["weights"], axis=0)
        shard_idx = manifests[name]["shards"]
        shard_path = cache_dir / f"{name}.shard{shard_idx:04d}.npz"
        counts = _save_shard(shard_path, x=x, y=y, weights=weights, pair_arrays=buf["pairs"])
        manifests[name]["paths"].append(str(shard_path))
        manifests[name]["examples"] += counts["examples"]
        manifests[name]["pairs"] += counts["pairs"]
        manifests[name]["shards"] += 1
        buf["x"].clear()
        buf["y"].clear()
        buf["weights"].clear()
        buf["pairs"] = _empty_pair_buffers()
        buf["examples"] = 0
        buf["bytes"] = 0

    def _add_examples(name: str, *, x, y, weights, pairs):
        buf = buffers[name]
        offset = buf["examples"]
        buf["x"].append(x)
        buf["y"].append(y)
        buf["weights"].append(weights)
        buf["examples"] += int(x.shape[0])
        buf["bytes"] += int(x.nbytes + y.nbytes + weights.nbytes)
        _append_pairs_with_offset(buf["pairs"], pairs, example_offset=offset)
        if buf["bytes"] >= int(buffer_bytes_limit):
            _flush(name)

    for result in _iter_problem_results(
        grouped_items,
        action_space=action_space,
        neighborhood_spec=neighborhood_spec,
        target_temperature=target_temperature,
        missing_penalty=2.0,
        pair_min_score_gap=pair_min_score_gap,
        max_pairs_per_example=max_pairs_per_example,
        num_workers=num_workers,
        worker_start_method=worker_start_method,
        max_tasks_in_flight=max_tasks_in_flight,
        progress_callback=progress_callback,
    ):
        if result.think_examples:
            key = _problem_key_from_example(result.think_examples[0])
            shard_name = "train_think" if key in train_keys else "val_think"
            x, y, weights = _prepare_think_boundary_inputs(result.think_examples, text_vectorizer, slot_memory, boundary_spec)
            _add_examples(shard_name, x=x, y=y, weights=weights, pairs=result.think_pairs)
        if result.answer_examples:
            key = _problem_key_from_example(result.answer_examples[0])
            shard_name = "train_answer" if key in train_keys else "val_answer"
            x, y, weights = _prepare_answer_boundary_inputs(result.answer_examples, text_vectorizer, slot_memory, boundary_spec)
            _add_examples(shard_name, x=x, y=y, weights=weights, pairs=result.answer_pairs)

    for name in manifests:
        _flush(name)
    return manifests, neighborhood_spec


def _load_shard(path: str):
    payload = np.load(path)
    return {
        "x": payload["x"],
        "y": payload["y"],
        "weights": payload["weights"],
        "pair_example_index": payload["pair_example_index"],
        "pair_winner_index": payload["pair_winner_index"],
        "pair_loser_index": payload["pair_loser_index"],
        "pair_weight": payload["pair_weight"],
    }


def _evaluate_boundary_from_shards(
    model: MLP,
    *,
    shard_paths: Sequence[str],
) -> Dict[str, float]:
    total_examples = 0
    total_dist = 0.0
    total_top1 = 0.0
    total_pairs = 0
    total_pair_correct = 0.0
    for path in shard_paths:
        shard = _load_shard(path)
        if shard["x"].shape[0] == 0:
            continue
        x_t = _to_tensor(shard["x"])
        y_t = _to_tensor(shard["y"])
        w_t = _to_tensor(shard["weights"] / np.maximum(1e-6, shard["weights"].mean()))
        with torch.no_grad():
            logits = model(x_t)
            log_probs = torch.log_softmax(logits, dim=-1)
            losses = -(y_t * log_probs).sum(dim=-1)
            total_dist += float((losses * w_t).sum().item())
            preds = torch.argmax(logits, dim=-1)
            labels = torch.argmax(y_t, dim=-1)
            total_top1 += float((preds == labels).float().sum().item())
            total_examples += int(x_t.shape[0])
            if shard["pair_example_index"].size > 0:
                pair_row_t = torch.tensor(shard["pair_example_index"], dtype=torch.long)
                winner_t = torch.tensor(shard["pair_winner_index"], dtype=torch.long)
                loser_t = torch.tensor(shard["pair_loser_index"], dtype=torch.long)
                pair_logits = logits[pair_row_t]
                winner_scores = pair_logits.gather(1, winner_t[:, None]).squeeze(1)
                loser_scores = pair_logits.gather(1, loser_t[:, None]).squeeze(1)
                total_pair_correct += float((winner_scores > loser_scores).float().sum().item())
                total_pairs += int(pair_row_t.shape[0])
    return {
        "distribution_loss": (total_dist / max(1, total_examples)),
        "top1_match": (total_top1 / max(1, total_examples)),
        "pairwise_accuracy": (total_pair_correct / max(1, total_pairs)),
    }


def _train_boundary_from_shards(
    model: MLP,
    *,
    shard_paths: Sequence[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    pairwise_weight: float,
    log_every: int,
    log_prefix: str,
) -> Dict[str, float]:
    if not shard_paths:
        return {"distribution_loss": 0.0, "top1_match": 0.0, "pairwise_accuracy": 0.0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start = time.perf_counter()
    for epoch in range(int(epochs)):
        epoch_shards = list(shard_paths)
        random.shuffle(epoch_shards)
        dist_sum = 0.0
        rank_sum = 0.0
        dist_den = 0
        rank_den = 0
        for path in epoch_shards:
            shard = _load_shard(path)
            if shard["x"].shape[0] == 0:
                continue
            x_t = _to_tensor(shard["x"])
            y_t = _to_tensor(shard["y"])
            w_t = _to_tensor(shard["weights"] / np.maximum(1e-6, shard["weights"].mean()))
            optimizer.zero_grad()
            logits = model(x_t)
            log_probs = torch.log_softmax(logits, dim=-1)
            dist_loss = (-(y_t * log_probs).sum(dim=-1) * w_t).mean()
            if shard["pair_example_index"].size > 0:
                pair_row_t = torch.tensor(shard["pair_example_index"], dtype=torch.long)
                winner_t = torch.tensor(shard["pair_winner_index"], dtype=torch.long)
                loser_t = torch.tensor(shard["pair_loser_index"], dtype=torch.long)
                pair_w_t = _to_tensor(shard["pair_weight"] / np.maximum(1e-6, shard["pair_weight"].mean()))
                pair_logits = logits[pair_row_t]
                rank_loss = pairwise_logistic_loss(pair_logits, winner_t, loser_t, pair_w_t)
                loss = dist_loss + (float(pairwise_weight) * rank_loss)
                rank_sum += float(rank_loss.item()) * int(pair_row_t.shape[0])
                rank_den += int(pair_row_t.shape[0])
            else:
                rank_loss = torch.tensor(0.0, dtype=dist_loss.dtype)
                loss = dist_loss
            loss.backward()
            optimizer.step()
            dist_sum += float(dist_loss.item()) * int(x_t.shape[0])
            dist_den += int(x_t.shape[0])
            del shard, x_t, y_t, w_t, logits, log_probs
        if int(log_every) > 0 and (((epoch + 1) % int(log_every) == 0) or (epoch + 1 == int(epochs))):
            _log(
                f"  [{log_prefix}] epoch {epoch + 1}/{int(epochs)} "
                f"dist_loss={(dist_sum / max(1, dist_den)):.4f} "
                f"rank_loss={(rank_sum / max(1, rank_den)):.4f} "
                f"elapsed={_format_seconds(time.perf_counter() - start)}"
            )
    return _evaluate_boundary_from_shards(model, shard_paths=shard_paths)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    overall_start = time.perf_counter()

    supervision_dir = Path(args.supervision_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    _log("[1/8] Loading offline supervision files...")
    scored_traces = read_jsonl(supervision_dir / "offline_supervision" / "scored_traces.jsonl")
    boundary_samples = read_jsonl(supervision_dir / "offline_supervision" / "boundary_samples.jsonl")
    boundary_pairs = read_jsonl(supervision_dir / "offline_supervision" / "boundary_preference_pairs.jsonl")
    _log(
        f"Loaded scored={len(scored_traces)} boundary_samples={len(boundary_samples)} "
        f"boundary_pairs={len(boundary_pairs)} in {_format_seconds(time.perf_counter() - stage_start)}"
    )

    action_space = DEFAULT_ACTION_SPACE
    neighborhood_config = _override_neighborhood_config(args)

    stage_start = time.perf_counter()
    _log("[2/8] Building prior examples...")
    prior_examples = build_prior_examples(scored_traces, action_space=action_space, target_temperature=float(args.target_temperature))
    _log(f"Built prior examples: {len(prior_examples)} in {_format_seconds(time.perf_counter() - stage_start)}")

    train_keys, val_keys = _split_problem_keys(
        prior_examples,
        seed=int(args.seed),
        val_frac=float(args.val_problem_frac),
        disable_val=bool(args.disable_val),
    )
    _log(f"[3/8] Problem split: train={len(train_keys)} val={len(val_keys)} (val disabled={bool(args.disable_val)})")

    train_prior_examples = _filter_prior_examples(prior_examples, train_keys)
    val_prior_examples = _filter_prior_examples(prior_examples, val_keys)

    text_vectorizer = HashTextVectorizer(dim=int(args.text_dim))
    prior_prompt_x = text_vectorizer.transform([row.prompt for row in train_prior_examples])
    prior_target_y = np.stack([row.action_distribution for row in train_prior_examples], axis=0).astype(np.float32)
    prior_weights = np.asarray([row.weight for row in train_prior_examples], dtype=np.float32)

    stage_start = time.perf_counter()
    _log("[4/8] Fitting slot memory...")
    slot_memory = SlotMemory.fit(
        prior_prompt_x,
        prior_target_y,
        num_slots=int(args.num_slots),
        seed=int(args.seed),
        temperature=float(args.slot_temperature),
        example_weights=prior_weights,
    )
    _log(f"Slot memory fit in {_format_seconds(time.perf_counter() - stage_start)}")

    train_boundary_rows = [row for row in boundary_samples if _problem_key_from_raw(row) in train_keys]
    boundary_spec = BoundaryFeatureSpec.fit(train_boundary_rows if train_boundary_rows else boundary_samples)

    cache_dir = output_dir / "boundary_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    buffer_bytes = _determine_boundary_buffer_bytes(args)

    stage_start = time.perf_counter()
    _log(
        "[5/8] Materializing boundary shards... "
        f"(buffer={buffer_bytes // (1024 * 1024)} MB, workers={int(args.num_workers)})"
    )
    progress_state = {"last_tick": -1}

    def _boundary_progress(done: int, total: int) -> None:
        if total <= 0:
            return
        tick = int((50 * done) / total)
        if done != total and tick == progress_state["last_tick"]:
            return
        progress_state["last_tick"] = tick
        filled = "#" * tick
        empty = "." * (50 - tick)
        pct = 100.0 * done / total
        print(f"\r  neighborhood [{filled}{empty}] {done}/{total} ({pct:5.1f}%)", end="", flush=True)
        if done == total:
            print("", flush=True)

    shard_manifest, neighborhood_spec = _materialize_boundary_shards(
        boundary_samples=boundary_samples,
        train_keys=train_keys,
        val_keys=val_keys,
        action_space=action_space,
        neighborhood_config=neighborhood_config,
        target_temperature=float(args.target_temperature),
        pair_min_score_gap=float(args.neighborhood_pair_gap),
        max_pairs_per_example=int(args.neighborhood_max_pairs),
        text_vectorizer=text_vectorizer,
        slot_memory=slot_memory,
        boundary_spec=boundary_spec,
        cache_dir=cache_dir,
        progress_callback=_boundary_progress,
        num_workers=int(args.num_workers),
        worker_start_method=str(args.worker_start_method),
        max_tasks_in_flight=(None if int(args.max_tasks_in_flight) <= 0 else int(args.max_tasks_in_flight)),
        buffer_bytes_limit=buffer_bytes,
    )
    _log(
        "Materialized boundary shards in "
        f"{_format_seconds(time.perf_counter() - stage_start)} "
        f"| train_think_shards={shard_manifest['train_think']['shards']} "
        f"| train_answer_shards={shard_manifest['train_answer']['shards']}"
    )

    stage_start = time.perf_counter()
    _log("[6/8] Preparing prior tensors...")
    prior_x, prior_y, prior_w = _prepare_prior_inputs(train_prior_examples, text_vectorizer, slot_memory)
    prior_val_x, prior_val_y, prior_val_w = _prepare_prior_inputs(val_prior_examples, text_vectorizer, slot_memory)
    _log(f"Prepared prior tensors in {_format_seconds(time.perf_counter() - stage_start)}")

    prior_model = MLP(input_dim=int(prior_x.shape[1]), hidden_dim=int(args.hidden_dim), output_dim=action_space.size, dropout=float(args.dropout))
    think_input_dim = int(text_vectorizer.dim) + action_space.size + boundary_spec.dim + action_space.size
    answer_input_dim = int(text_vectorizer.dim) + action_space.size + boundary_spec.dim + action_space.info_size
    think_boundary_model = MLP(input_dim=think_input_dim, hidden_dim=int(args.hidden_dim), output_dim=action_space.size, dropout=float(args.dropout))
    answer_boundary_model = MLP(input_dim=answer_input_dim, hidden_dim=int(args.hidden_dim), output_dim=action_space.info_size, dropout=float(args.dropout))

    stage_start = time.perf_counter()
    _log("[7/8] Training prior model...")
    prior_metrics = train_prior_model(
        prior_model,
        x=prior_x,
        y=prior_y,
        weights=prior_w,
        epochs=int(args.prior_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        log_every=int(args.prior_log_every),
        log_prefix="prior",
    )
    prior_val_metrics = evaluate_prior_model(prior_model, x=prior_val_x, y=prior_val_y, weights=prior_val_w)
    _log(
        f"Prior done in {_format_seconds(time.perf_counter() - stage_start)} "
        f"| train_top1={prior_metrics['top1_match']:.3f} "
        f"| val_top1={prior_val_metrics['top1_match']:.3f}"
    )
    del prior_x, prior_y, prior_w, prior_val_x, prior_val_y, prior_val_w

    stage_start = time.perf_counter()
    _log("[8/8] Training think/answer boundary models from shards...")
    think_boundary_metrics = _train_boundary_from_shards(
        think_boundary_model,
        shard_paths=shard_manifest["train_think"]["paths"],
        epochs=int(args.think_boundary_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pairwise_weight=float(args.pairwise_weight),
        log_every=int(args.boundary_log_every),
        log_prefix="think_boundary",
    )
    think_boundary_val_metrics = _evaluate_boundary_from_shards(
        think_boundary_model,
        shard_paths=shard_manifest["val_think"]["paths"],
    )
    answer_boundary_metrics = _train_boundary_from_shards(
        answer_boundary_model,
        shard_paths=shard_manifest["train_answer"]["paths"],
        epochs=int(args.answer_boundary_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pairwise_weight=float(args.pairwise_weight),
        log_every=int(args.boundary_log_every),
        log_prefix="answer_boundary",
    )
    answer_boundary_val_metrics = _evaluate_boundary_from_shards(
        answer_boundary_model,
        shard_paths=shard_manifest["val_answer"]["paths"],
    )
    _log(
        f"Boundary models done in {_format_seconds(time.perf_counter() - stage_start)} "
        f"| think_val_pair_acc={think_boundary_val_metrics['pairwise_accuracy']:.3f} "
        f"| answer_val_pair_acc={answer_boundary_val_metrics['pairwise_accuracy']:.3f}"
    )

    artifact = {
        "seed": int(args.seed),
        "action_space": {
            "info_values": list(action_space.info_values),
            "cot_values": list(action_space.cot_values),
            "cot_token_budgets": dict(action_space.cot_token_budgets),
        },
        "text_vectorizer": asdict(text_vectorizer),
        "slot_memory": slot_memory.to_dict(),
        "boundary_spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in boundary_spec.__dict__.items()},
        "boundary_neighborhood": neighborhood_spec.to_dict(),
        "prior_model": {
            "input_dim": int(int(text_vectorizer.dim) + action_space.size),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(action_space.size),
            "state_dict": {k: v.cpu() for k, v in prior_model.state_dict().items()},
        },
        "think_boundary_model": {
            "input_dim": int(think_input_dim),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(action_space.size),
            "state_dict": {k: v.cpu() for k, v in think_boundary_model.state_dict().items()},
        },
        "answer_boundary_model": {
            "input_dim": int(answer_input_dim),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(action_space.info_size),
            "state_dict": {k: v.cpu() for k, v in answer_boundary_model.state_dict().items()},
        },
        "train_metrics": {
            "prior": {"train": prior_metrics, "val": prior_val_metrics},
            "think_boundary": {"train": think_boundary_metrics, "val": think_boundary_val_metrics},
            "answer_boundary": {"train": answer_boundary_metrics, "val": answer_boundary_val_metrics},
        },
        "counts": {
            "scored_traces": len(scored_traces),
            "boundary_samples": len(boundary_samples),
            "boundary_preference_pairs_input": len(boundary_pairs),
            "prior_examples": len(prior_examples),
            "train_prior_examples": len(train_prior_examples),
            "val_prior_examples": len(val_prior_examples),
            "train_think_boundary_examples": shard_manifest["train_think"]["examples"],
            "val_think_boundary_examples": shard_manifest["val_think"]["examples"],
            "train_think_boundary_pairs": shard_manifest["train_think"]["pairs"],
            "val_think_boundary_pairs": shard_manifest["val_think"]["pairs"],
            "train_answer_boundary_examples": shard_manifest["train_answer"]["examples"],
            "val_answer_boundary_examples": shard_manifest["val_answer"]["examples"],
            "train_answer_boundary_pairs": shard_manifest["train_answer"]["pairs"],
            "val_answer_boundary_pairs": shard_manifest["val_answer"]["pairs"],
            "train_problem_count": len(train_keys),
            "val_problem_count": len(val_keys),
            "train_think_shards": shard_manifest["train_think"]["shards"],
            "val_think_shards": shard_manifest["val_think"]["shards"],
            "train_answer_shards": shard_manifest["train_answer"]["shards"],
            "val_answer_shards": shard_manifest["val_answer"]["shards"],
        },
    }

    torch.save(artifact, output_dir / "autocrat_controller.pt")
    write_json(
        output_dir / "training_summary.json",
        {
            "supervision_dir": str(supervision_dir),
            "output_dir": str(output_dir),
            "counts": artifact["counts"],
            "train_metrics": artifact["train_metrics"],
            "slot_count": len(slot_memory.centers),
            "action_labels": [action_space.action_label(i) for i in range(action_space.size)],
            "answer_info_labels": [f"i{info_mode}" for info_mode in action_space.info_values],
            "boundary_neighborhood": artifact["boundary_neighborhood"],
            "boundary_cache_dir": str(cache_dir),
            "boundary_buffer_mb": int(buffer_bytes // (1024 * 1024)),
            "elapsed": _format_seconds(time.perf_counter() - overall_start),
        },
    )
    if not bool(args.keep_boundary_cache):
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "counts": artifact["counts"],
                "train_metrics": artifact["train_metrics"],
                "slot_count": len(slot_memory.centers),
                "elapsed": _format_seconds(time.perf_counter() - overall_start),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
