#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import concatenate_datasets, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]

PAPER_SPLITS: Dict[str, Tuple[int, int]] = {
    "humaneval": (33, 131),
    "mbpp": (86, 341),
    "gsm8k": (264, 1055),
    "math_500": (100, 400),
    "arc_challenge": (458, 1833),
    "gpqa_diamond": (40, 158),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the six AutoCRAT divided benchmark files from public benchmark sources. "
            "The default counts match the paper appendix exactly."
        )
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "data" / "divided_benchmarks"),
        help="Directory that will receive one train.jsonl and one test.jsonl per benchmark.",
    )
    parser.add_argument("--seed", type=int, default=20260412, help="Deterministic split seed.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(PAPER_SPLITS),
        choices=sorted(PAPER_SPLITS),
        help="Optional benchmark allowlist.",
    )
    parser.add_argument(
        "--allow-truncate",
        action="store_true",
        help="Allow writing fewer examples if a local mirror is smaller than the paper split.",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _dataset_rows(dataset: Any) -> List[Dict[str, Any]]:
    return [dict(row) for row in dataset]


def _load_humaneval() -> List[Dict[str, Any]]:
    return _dataset_rows(load_dataset("openai/openai_humaneval", split="test"))


def _load_mbpp() -> List[Dict[str, Any]]:
    rows = _dataset_rows(load_dataset("mbpp", split="test"))
    if len(rows) < sum(PAPER_SPLITS["mbpp"]):
        rows = _dataset_rows(load_dataset("mbpp", split="train"))
    return rows


def _load_gsm8k() -> List[Dict[str, Any]]:
    return _dataset_rows(load_dataset("openai/gsm8k", "main", split="test"))


def _load_math_500() -> List[Dict[str, Any]]:
    return _dataset_rows(load_dataset("HuggingFaceH4/MATH-500", split="test"))


def _load_arc_challenge() -> List[Dict[str, Any]]:
    train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    return _dataset_rows(concatenate_datasets([train, test]))


def _load_gpqa_diamond() -> List[Dict[str, Any]]:
    return _dataset_rows(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))


LOADERS = {
    "humaneval": _load_humaneval,
    "mbpp": _load_mbpp,
    "gsm8k": _load_gsm8k,
    "math_500": _load_math_500,
    "arc_challenge": _load_arc_challenge,
    "gpqa_diamond": _load_gpqa_diamond,
}


def _stable_id(dataset: str, row: Dict[str, Any], index: int) -> str:
    for key in ("task_id", "id", "problem_id", "unique_id", "Record ID"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{dataset}_{index}"


def _enrich_rows(dataset: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        item = dict(row)
        item.setdefault("problem_id", _stable_id(dataset, item, idx))
        item["_autocrat_dataset"] = dataset
        enriched.append(item)
    return enriched


def _split_rows(
    dataset: str,
    rows: Sequence[Dict[str, Any]],
    *,
    seed: int,
    allow_truncate: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_count, test_count = PAPER_SPLITS[dataset]
    required = train_count + test_count
    if len(rows) < required and not allow_truncate:
        raise ValueError(
            f"{dataset} has {len(rows)} rows, but the paper split requires {required}. "
            "Use --allow-truncate only when working with a smaller local mirror."
        )
    take = min(len(rows), required)
    indices = list(range(len(rows)))
    rng = random.Random(f"{seed}:{dataset}")
    rng.shuffle(indices)
    selected = indices[:take]
    train_indices = set(selected[: min(train_count, take)])
    test_indices = set(selected[min(train_count, take) : min(required, take)])
    train_rows = [dict(rows[idx]) for idx in range(len(rows)) if idx in train_indices]
    test_rows = [dict(rows[idx]) for idx in range(len(rows)) if idx in test_indices]
    return _enrich_rows(dataset, train_rows), _enrich_rows(dataset, test_rows)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest: Dict[str, Any] = {
        "output_root": str(output_root),
        "seed": int(args.seed),
        "paper_splits": {
            key: {"train": train, "test": test}
            for key, (train, test) in sorted(PAPER_SPLITS.items())
            if key in set(args.datasets)
        },
        "datasets": {},
    }

    for dataset in args.datasets:
        rows = LOADERS[dataset]()
        train_rows, test_rows = _split_rows(
            dataset,
            rows,
            seed=int(args.seed),
            allow_truncate=bool(args.allow_truncate),
        )
        dataset_dir = output_root / dataset
        _write_jsonl(dataset_dir / "train.jsonl", train_rows)
        _write_jsonl(dataset_dir / "test.jsonl", test_rows)
        manifest["datasets"][dataset] = {
            "source_rows": len(rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_path": str(dataset_dir / "train.jsonl"),
            "test_path": str(dataset_dir / "test.jsonl"),
        }

    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
