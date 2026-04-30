#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


CODE_DATASETS = {"mbpp", "humaneval"}
RUN_RE = re.compile(r"benchmarks_tiered_(v5)_wide12_([^/]+)_run1")


@dataclass(frozen=True)
class MBPPItem:
    problem_id: str
    prompt: str
    tests: List[str]
    test_setup_code: str = ""


@dataclass(frozen=True)
class HumanEvalItem:
    task_id: str
    prompt: str
    test: str
    entry_point: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute real code evaluation for MBPP/HumanEval predictions.")
    parser.add_argument(
        "--root",
        default="runs",
        help="Root folder that contains benchmark run folders.",
    )
    parser.add_argument(
        "--phase-prefix",
        nargs="*",
        default=["v5"],
        help="Benchmark phases to scan (default: v5).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=4.0,
        help="Per-sample execution timeout seconds.",
    )
    parser.add_argument(
        "--python-bin",
        default="python",
        help="Python executable used for sandboxed execution.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input raw_predictions.jsonl with execution-correctness labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="../result",
        help="Directory for generated metrics CSV/JSON.",
    )
    return parser.parse_args()


_RE_THINK_BLOCKS = [
    re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|startofthink\|>.*?<\|endofthink\|>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", flags=re.IGNORECASE | re.DOTALL),
]
_RE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_RE_CODE_START = re.compile(
    r"^\s*(def |class |from |import |@|if __name__ ==|for |while |try:|with |return\b|assert\b|[A-Za-z_]\w*\s*=)"
)
_RE_CODE_START_ANY = re.compile(
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


def _strip_implicit_reasoning_prefix(s: str) -> str:
    text = s or ""
    lowered = text.lower()
    close_idx = lowered.rfind("</think>")
    if close_idx < 0:
        return text
    tail = text[close_idx + len("</think>") :].strip()
    if tail:
        return tail
    return text


def _strip_reasoning_noise(s: str) -> str:
    out = s
    for p in _RE_THINK_BLOCKS:
        out = p.sub(" ", out)
    out = _strip_implicit_reasoning_prefix(out)
    return out


def _normalize_candidate(code: str) -> str:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    cleaned: List[str] = []
    for line in lines:
        t = line.strip()
        if t in {"```", "python", "py"}:
            continue
        if t.startswith("FINAL_ANSWER:"):
            line = line[line.find(":") + 1 :].lstrip()
        cleaned.append(line.rstrip())
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _strip_leading_instruction_lines(s: str) -> str:
    lines = s.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or _RE_INSTR_LINE.match(lines[i])):
        i += 1
    return "\n".join(lines[i:]).strip()


def _looks_code_line(line: str) -> bool:
    t = line.rstrip()
    if not t.strip():
        return True
    if _RE_CODE_START.match(t):
        return True
    if t.lstrip().startswith("#"):
        return True
    if t.startswith((" ", "\t")):
        return True
    if t.strip().endswith((",", ":", ")", "]", "}", "\\")):
        return True
    if any(tok in t for tok in ["==", "!=", "<=", ">=", " and ", " or ", " in ", " not in ", " is "]):
        return True
    return False


def _candidate_from_first_code_line(s: str) -> str:
    lines = s.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _RE_CODE_START.match(line):
            start = i
            break
    if start is None:
        return ""
    cand = lines[start:]
    while cand and _RE_NARRATIVE.match(cand[-1]) and not _looks_code_line(cand[-1]):
        cand.pop()
    return "\n".join(cand).strip()


def _candidate_from_first_code_anchor(s: str) -> str:
    m = _RE_CODE_START_ANY.search(s)
    if not m:
        return ""
    return s[m.start() :].strip()


def _candidate_code_like_only(s: str) -> str:
    lines = s.splitlines()
    kept: List[str] = []
    started = False
    for line in lines:
        if _looks_code_line(line):
            kept.append(line)
            started = True
            continue
        if started and _RE_NARRATIVE.match(line):
            # likely trailing explanation after code block
            break
        if started:
            kept.append(line)
    return "\n".join(kept).strip()


def _score_candidate(code: str) -> Tuple[int, int, int, int]:
    c = _normalize_candidate(code)
    if not c:
        return (0, 0, 0, 0)
    parse_ok = 0
    try:
        ast.parse(c)
        parse_ok = 1
    except Exception:
        parse_ok = 0
    has_core = 1 if re.search(r"^\s*(def |class |from |import )", c, flags=re.MULTILINE) else 0
    non_empty = sum(1 for ln in c.splitlines() if ln.strip())
    return (parse_ok, has_core, non_empty, len(c))


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


def _extract_expected_callable_names_from_mbpp_tests(tests: List[str]) -> List[str]:
    names: List[str] = []
    pat = re.compile(
        r"""assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|==|!=|<=|>=|<|>|is\b|in\b)""",
        flags=re.IGNORECASE,
    )
    for t in tests:
        for m in pat.finditer(t):
            name = m.group(1)
            if name and name not in names:
                names.append(name)
    return names


def _append_missing_aliases(code: str, expected_names: List[str]) -> str:
    if not expected_names:
        return code
    top_level = _top_level_callable_names(code)
    if not top_level:
        return code
    present = set(top_level)
    missing = [n for n in expected_names if n not in present]
    if not missing:
        return code

    # Conservative aliasing: only when there is a single generated callable.
    if len(top_level) != 1:
        return code
    src = top_level[0]
    alias_lines = [f"{n} = {src}" for n in missing if n != src]
    if not alias_lines:
        return code
    return code.rstrip() + "\n\n" + "\n".join(alias_lines) + "\n"


def _extract_code(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = _strip_reasoning_noise(s)
    s = _strip_leading_instruction_lines(s)

    candidates: List[str] = []
    candidates.append(s)

    lowered = s.lower()
    close_idx = lowered.rfind("</think>")
    if close_idx >= 0:
        after_close = s[close_idx + len("</think>") :].strip()
        if after_close:
            candidates.append(after_close)

    fenced = _RE_FENCE.findall(s)
    candidates.extend(fenced)
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
    seen = set()
    for cand in candidates:
        norm = _normalize_candidate(cand)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        score = _score_candidate(norm)
        if score > best_score:
            best_score = score
            best = norm
    return best


def _run_python(script: str, *, timeout_sec: float, python_bin: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="autocrat_code_eval_") as tmp:
        p = Path(tmp) / "run_eval.py"
        p.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [python_bin, str(p)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or proc.stdout or "").strip()
    return False, err[:300]


def _load_mbpp() -> Dict[str, MBPPItem]:
    # Prefer canonical MBPP source with stable `test_list` and split ordering
    # aligned with our `mbpp_{index}` problem IDs.
    ds = None
    for args in (
        ("google-research-datasets/mbpp",),
        ("mbpp",),
        ("mbpp", "sanitized"),
    ):
        try:
            ds = load_dataset(*args, split="test")
            break
        except Exception:
            ds = None
    if ds is None:
        raise RuntimeError("failed to load MBPP dataset from all known sources")

    items: Dict[str, MBPPItem] = {}
    for i, ex in enumerate(ds):
        tests = [str(x) for x in ex.get("test_list", [])]
        prompt = str(ex.get("prompt", ex.get("text", ""))).strip()
        item = MBPPItem(
            problem_id=f"mbpp_{i}",
            prompt=prompt,
            tests=tests,
            test_setup_code=str(ex.get("test_setup_code", "") or ""),
        )
        items[f"mbpp_{i}"] = item
        task_id = ex.get("task_id")
        if task_id is not None:
            items[f"mbpp_{task_id}"] = item
    return items


def _load_humaneval() -> Dict[str, HumanEvalItem]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    items: Dict[str, HumanEvalItem] = {}
    for ex in ds:
        task_id = str(ex["task_id"])
        items[task_id] = HumanEvalItem(
            task_id=task_id,
            prompt=str(ex["prompt"]),
            test=str(ex["test"]),
            entry_point=str(ex["entry_point"]),
        )
    return items


def _eval_mbpp(code: str, item: MBPPItem, *, timeout_sec: float, python_bin: str) -> Tuple[bool, str]:
    if not code.strip():
        return False, "empty_code"
    expected_names = _extract_expected_callable_names_from_mbpp_tests(item.tests)
    prepared = _append_missing_aliases(code, expected_names)
    test_block = "\n".join(item.tests)
    script_parts = [prepared]
    if str(item.test_setup_code or "").strip():
        script_parts.append(str(item.test_setup_code))
    script_parts.append(test_block)
    script = "\n\n".join(script_parts)
    return _run_python(script, timeout_sec=timeout_sec, python_bin=python_bin)


def _eval_humaneval(code: str, item: HumanEvalItem, *, timeout_sec: float, python_bin: str) -> Tuple[bool, str]:
    completion = code.strip()
    if not completion:
        return False, "empty_code"

    completion = _append_missing_aliases(completion, [item.entry_point])

    variants: List[str] = [completion]
    if not completion.startswith(item.prompt.strip()):
        prompt_completion = item.prompt + completion
        prompt_completion = _append_missing_aliases(prompt_completion, [item.entry_point])
        variants.append(prompt_completion)

    for candidate in variants:
        script = "\n\n".join([candidate, item.test, f"check({item.entry_point})"])
        ok, err = _run_python(script, timeout_sec=timeout_sec, python_bin=python_bin)
        if ok:
            return True, ""
    return False, err


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _norm_prompt(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    if k > n:
        k = n
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _scan_inputs(root: Path, phase_prefixes: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.glob("benchmarks_tiered_*_wide12_*_run1/raw_predictions.jsonl")):
        m = RUN_RE.search(str(p.parent))
        if not m:
            continue
        phase = m.group(1)
        if phase in phase_prefixes:
            out.append(p)
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mbpp = _load_mbpp()
    mbpp_by_prompt: Dict[str, MBPPItem] = {}
    for it in mbpp.values():
        key = _norm_prompt(it.prompt)
        if key and key not in mbpp_by_prompt:
            mbpp_by_prompt[key] = it
    humaneval = _load_humaneval()

    inputs = _scan_inputs(root, args.phase_prefix)
    if not inputs:
        print("no input files found")
        return 1

    per_config_rows: List[Dict[str, object]] = []
    per_model_problem: Dict[Tuple[str, str, str, str], List[int]] = {}
    file_stats: List[Dict[str, object]] = []

    for path in inputs:
        m = RUN_RE.search(str(path.parent))
        if not m:
            continue
        phase = m.group(1)
        model_key = m.group(2)
        rows = _read_jsonl(path)
        total_code = 0
        fixed = 0

        for row in rows:
            dataset = str(row.get("dataset", ""))
            if dataset not in CODE_DATASETS:
                continue
            total_code += 1
            problem_id = str(row.get("problem_id", ""))
            code = _extract_code(str(row.get("prediction", "")))
            ok = False
            err = ""
            if dataset == "mbpp":
                item = mbpp.get(problem_id)
                if item is None:
                    item = mbpp_by_prompt.get(_norm_prompt(str(row.get("prompt", ""))))
                if item is None:
                    err = "missing_mbpp_problem"
                else:
                    ok, err = _eval_mbpp(code, item, timeout_sec=float(args.timeout_sec), python_bin=args.python_bin)
            elif dataset == "humaneval":
                item = humaneval.get(problem_id)
                if item is None:
                    err = "missing_humaneval_problem"
                else:
                    ok, err = _eval_humaneval(
                        code,
                        item,
                        timeout_sec=float(args.timeout_sec),
                        python_bin=args.python_bin,
                    )
            row["is_correct_exec"] = 1.0 if ok else 0.0
            if dataset in CODE_DATASETS:
                row["is_correct"] = row["is_correct_exec"]
                fixed += 1
            if err:
                row["exec_error"] = err

            cfg = f"i{int(row.get('info_mode', 0))}_cot{int(row.get('cot_mode', 0))}"
            key = (phase, model_key, dataset, cfg)
            per_model_problem.setdefault((phase, model_key, dataset, problem_id), [])
            per_model_problem[(phase, model_key, dataset, problem_id)].append(int(bool(ok)))

        if args.in_place:
            _write_jsonl(path, rows)
        else:
            out_path = out_dir / f"{path.parent.name}.raw_predictions.exec.jsonl"
            _write_jsonl(out_path, rows)

        # per-config pass@1 summary (from execution labels)
        group: Dict[Tuple[str, str], List[int]] = {}
        for row in rows:
            dataset = str(row.get("dataset", ""))
            if dataset not in CODE_DATASETS:
                continue
            cfg = f"i{int(row.get('info_mode', 0))}_cot{int(row.get('cot_mode', 0))}"
            group.setdefault((dataset, cfg), []).append(int(float(row.get("is_correct_exec", 0.0)) > 0.0))

        for (dataset, cfg), vals in sorted(group.items()):
            c = sum(vals)
            n = len(vals)
            per_config_rows.append(
                {
                    "phase": phase,
                    "model_key": model_key,
                    "dataset": dataset,
                    "config_mode": cfg,
                    "n": n,
                    "correct": c,
                    "pass_at_1": (c / n) if n else 0.0,
                }
            )

        file_stats.append(
            {
                "file": str(path),
                "phase": phase,
                "model_key": model_key,
                "code_rows": total_code,
                "updated_rows": fixed,
            }
        )

    # cross-config pass@k (k over attempts from all configs in same phase/model/dataset)
    cross_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str, str], List[List[int]]] = {}
    for (phase, model_key, dataset, _problem_id), vals in per_model_problem.items():
        grouped.setdefault((phase, model_key, dataset), []).append(vals)

    for (phase, model_key, dataset), attempts_per_problem in sorted(grouped.items()):
        for k in (1, 3, 5, 10):
            score = 0.0
            n_probs = len(attempts_per_problem)
            for attempts in attempts_per_problem:
                n = len(attempts)
                c = sum(attempts)
                score += _pass_at_k(n, c, k)
            cross_rows.append(
                {
                    "phase": phase,
                    "model_key": model_key,
                    "dataset": dataset,
                    "k": k,
                    "num_problems": n_probs,
                    "pass_at_k_cross_config": (score / n_probs) if n_probs else 0.0,
                }
            )

    per_config_csv = out_dir / "code_exec_pass_at_1_by_config.csv"
    with per_config_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["phase", "model_key", "dataset", "config_mode", "n", "correct", "pass_at_1"],
        )
        w.writeheader()
        for row in per_config_rows:
            w.writerow(row)

    cross_csv = out_dir / "code_exec_pass_at_k_cross_config.csv"
    with cross_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["phase", "model_key", "dataset", "k", "num_problems", "pass_at_k_cross_config"],
        )
        w.writeheader()
        for row in cross_rows:
            w.writerow(row)

    summary_path = out_dir / "code_exec_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "inputs": [str(p) for p in inputs],
                "file_stats": file_stats,
                "outputs": {
                    "pass_at_1_by_config": str(per_config_csv),
                    "pass_at_k_cross_config": str(cross_csv),
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"processed_files": len(inputs), "summary": str(summary_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
