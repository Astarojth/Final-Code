#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark eval items jsonl for baseline runs.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/benchmarks/eval_items.benchmarks_tiered_v5.jsonl"),
        help="Output eval items jsonl path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--main-gsm8k",
        type=int,
        default=25,
        help="Main layer size for GSM8K.",
    )
    parser.add_argument(
        "--main-gsm8k-easy",
        type=int,
        default=20,
        help="Main layer size for GSM8K easy subset (short-chain friendly).",
    )
    parser.add_argument(
        "--main-boolq",
        type=int,
        default=25,
        help="Main layer size for BoolQ.",
    )
    parser.add_argument(
        "--main-arc-easy",
        type=int,
        default=25,
        help="Main layer size for ARC-Easy (short-chain friendly).",
    )
    parser.add_argument(
        "--main-arc-challenge",
        type=int,
        default=25,
        help="Main layer size for ARC-Challenge.",
    )
    parser.add_argument(
        "--main-commonsenseqa",
        type=int,
        default=25,
        help="Main layer size for CommonsenseQA.",
    )
    parser.add_argument(
        "--main-openbookqa",
        type=int,
        default=25,
        help="Main layer size for OpenBookQA (medium difficulty, short-chain friendly).",
    )
    parser.add_argument(
        "--main-mbpp",
        type=int,
        default=25,
        help="Main layer size for MBPP easy subset.",
    )
    parser.add_argument(
        "--challenge-humaneval",
        type=int,
        default=10,
        help="Challenge layer size for HumanEval.",
    )
    parser.add_argument(
        "--challenge-math-hard",
        type=int,
        default=10,
        help="Challenge layer size for hard MATH subset.",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _extract_math_reference(solution: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if boxed:
        return boxed[-1].strip()
    if "####" in solution:
        return solution.split("####")[-1].strip()
    return solution.strip()


def _sample_indices(total: int, take: int, rng: random.Random) -> List[int]:
    take = min(take, total)
    idx = list(range(total))
    rng.shuffle(idx)
    return sorted(idx[:take])


def _build_gsm8k(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i in _sample_indices(len(ds), limit, rng):
        ex = ds[i]
        rows.append(
            {
                "dataset": "gsm8k",
                "problem_id": f"gsm8k_{i}",
                "category": "math",
                "prompt": ex["question"],
                "reference": ex["answer"],
            }
        )
    return rows


def _gsm8k_complexity(question: str, answer: str) -> int:
    op_cnt = len(re.findall(r"[+\-*/]", answer))
    step_cnt = answer.count("\n")
    q_len = len(question)
    return (op_cnt * 8) + (step_cnt * 4) + (q_len // 32)


def _build_gsm8k_easy(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ranked = []
    for i, ex in enumerate(ds):
        question = str(ex["question"])
        answer = str(ex["answer"])
        ranked.append((_gsm8k_complexity(question, answer), i, ex))
    ranked.sort(key=lambda x: x[0])

    pool_size = min(len(ranked), max(limit * 6, limit))
    pool = ranked[:pool_size]
    rng.shuffle(pool)
    chosen = sorted(pool[: min(limit, len(pool))], key=lambda x: x[1])

    rows = []
    for _, i, ex in chosen:
        rows.append(
            {
                "dataset": "gsm8k_easy",
                "problem_id": f"gsm8k_easy_{i}",
                "category": "math",
                "prompt": ex["question"],
                "reference": ex["answer"],
            }
        )
    return rows


def _build_math_hard(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    # Challenge subset: level 3/4 from challenging configs (reduced from 4/5).
    configs = ["intermediate_algebra", "number_theory", "precalculus"]
    pools = []
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        hard = [ex for ex in ds if str(ex.get("level", "")).strip() in {"Level 3", "Level 4"}]
        for ex in hard:
            pools.append((cfg, ex))

    rng.shuffle(pools)
    pools = pools[: min(limit, len(pools))]
    rows = []
    for idx, (cfg, ex) in enumerate(pools):
        rows.append(
            {
                "dataset": "math_hard",
                "problem_id": f"math_hard_{cfg}_{idx}",
                "category": "math",
                "prompt": ex["problem"],
                "reference": _extract_math_reference(ex["solution"]),
            }
        )
    return rows


def _render_arc_prompt(question: str, choices: Dict[str, List[str]]) -> str:
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    lines = [question, ""]
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")
    lines.append("")
    lines.append("Answer with one choice letter.")
    return "\n".join(lines)


def _build_arc_easy(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    rows = []
    for i in _sample_indices(len(ds), limit, rng):
        ex = ds[i]
        rows.append(
            {
                "dataset": "arc_easy",
                "problem_id": f"arc_easy_{i}",
                "category": "logic",
                "prompt": _render_arc_prompt(str(ex["question"]), dict(ex["choices"])),
                "reference": str(ex["answerKey"]).strip(),
            }
        )
    return rows


def _build_arc_challenge(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i in _sample_indices(len(ds), limit, rng):
        ex = ds[i]
        rows.append(
            {
                "dataset": "arc_challenge",
                "problem_id": f"arc_challenge_{i}",
                "category": "logic",
                "prompt": _render_arc_prompt(str(ex["question"]), dict(ex["choices"])),
                "reference": str(ex["answerKey"]).strip(),
            }
        )
    return rows


def _build_commonsenseqa(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("tau/commonsense_qa", split="validation")
    rows = []
    for i in _sample_indices(len(ds), limit, rng):
        ex = ds[i]
        rows.append(
            {
                "dataset": "commonsenseqa",
                "problem_id": str(ex.get("id", f"commonsenseqa_{i}")),
                "category": "logic",
                "prompt": _render_arc_prompt(str(ex["question"]), dict(ex["choices"])),
                "reference": str(ex["answerKey"]).strip(),
            }
        )
    return rows


def _build_openbookqa(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    rows = []
    for i in _sample_indices(len(ds), limit, rng):
        ex = ds[i]
        prompt = _render_arc_prompt(str(ex["question_stem"]), dict(ex["choices"]))
        rows.append(
            {
                "dataset": "openbookqa",
                "problem_id": str(ex.get("id", f"openbookqa_{i}")),
                "category": "logic",
                "prompt": prompt,
                "reference": str(ex["answerKey"]).strip(),
            }
        )
    return rows


def _build_mbpp(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("mbpp", split="test")
    ranked = []
    for i, ex in enumerate(ds):
        prompt = str(ex["text"])
        code = str(ex["code"])
        line_count = code.count("\n") + 1
        # Heuristic easy-score: shorter prompt + shorter canonical code + fewer lines.
        easy_score = len(prompt) + len(code) + (line_count * 20)
        ranked.append((easy_score, i, ex))

    ranked.sort(key=lambda x: x[0])
    pool_size = min(len(ranked), max(limit * 4, limit))
    pool = ranked[:pool_size]
    rng.shuffle(pool)
    chosen = sorted(pool[: min(limit, len(pool))], key=lambda x: x[1])

    rows = []
    for _, i, ex in chosen:
        rows.append(
            {
                "dataset": "mbpp",
                "problem_id": f"mbpp_{i}",
                "category": "code",
                "prompt": ex["text"],
                "reference": ex["code"],
            }
        )
    return rows


def _build_humaneval(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for i in _sample_indices(len(ds), limit, rng):
        ex = ds[i]
        rows.append(
            {
                "dataset": "humaneval",
                "problem_id": ex["task_id"],
                "category": "code",
                "prompt": ex["prompt"],
                "reference": ex["canonical_solution"],
            }
        )
    return rows


def _build_boolq(limit: int, rng: random.Random) -> List[Dict[str, str]]:
    ds = load_dataset("google/boolq", split="validation")
    rows = []
    for i in _sample_indices(len(ds), limit, rng):
        ex = ds[i]
        prompt = (
            "Read the passage and answer the yes/no question.\n\n"
            f"Passage:\n{ex['passage']}\n\n"
            f"Question:\n{ex['question']}\n"
        )
        rows.append(
            {
                "dataset": "boolq",
                "problem_id": f"boolq_{i}",
                "category": "logic",
                "prompt": prompt,
                "reference": "true" if bool(ex["answer"]) else "false",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = []
    # Main layer.
    rows.extend(_build_gsm8k(args.main_gsm8k, rng))
    rows.extend(_build_gsm8k_easy(args.main_gsm8k_easy, rng))
    rows.extend(_build_boolq(args.main_boolq, rng))
    rows.extend(_build_arc_easy(args.main_arc_easy, rng))
    rows.extend(_build_arc_challenge(args.main_arc_challenge, rng))
    rows.extend(_build_commonsenseqa(args.main_commonsenseqa, rng))
    rows.extend(_build_openbookqa(args.main_openbookqa, rng))
    rows.extend(_build_mbpp(args.main_mbpp, rng))
    # Challenge layer.
    rows.extend(_build_humaneval(args.challenge_humaneval, rng))
    rows.extend(_build_math_hard(args.challenge_math_hard, rng))

    out = Path(args.output).expanduser().resolve()
    _write_jsonl(out, rows)
    print(
        json.dumps(
            {
                "output": str(out),
                "counts": {
                    "main": {
                        "gsm8k": args.main_gsm8k,
                        "gsm8k_easy": args.main_gsm8k_easy,
                        "boolq": args.main_boolq,
                        "arc_easy": args.main_arc_easy,
                        "arc_challenge": args.main_arc_challenge,
                        "commonsenseqa": args.main_commonsenseqa,
                        "openbookqa": args.main_openbookqa,
                        "mbpp_easy": args.main_mbpp,
                    },
                    "challenge": {
                        "humaneval": args.challenge_humaneval,
                        "math_hard": args.challenge_math_hard,
                    },
                    "total": len(rows),
                },
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
