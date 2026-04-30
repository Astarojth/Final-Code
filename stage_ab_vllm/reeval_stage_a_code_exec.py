#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocrat.offline_supervision import write_jsonl
from autocrat.trace_collection import collect_static_traces
from scripts.eval_code_exec import MBPPItem, _eval_mbpp, _extract_code
from stage_ab_vllm.stage_ab_vllm_pipeline import _aggregate_raw_rows, _prepare_offline_outputs


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _load_source_rows(source_path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(source_path)


def _extract_tests_from_source_row(source_row: Dict[str, Any]) -> List[str]:
    tests_raw = source_row.get("test_list", source_row.get("tests", []))
    if isinstance(tests_raw, list):
        return [str(x) for x in tests_raw if str(x).strip()]
    if isinstance(tests_raw, str) and tests_raw.strip():
        return [tests_raw]
    return []


def _hard_items_from_normalized_rows(
    normalized_rows: Sequence[Dict[str, Any]],
    hard_examples: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    keep = {
        (str(row.get("dataset", "")), str(row.get("problem_id", "")))
        for row in hard_examples
    }
    out: List[Dict[str, Any]] = []
    for row in normalized_rows:
        dataset = str(row.get("dataset", ""))
        problem_id = str(row.get("problem_id", ""))
        if (dataset, problem_id) not in keep:
            continue
        out.append(
            {
                "dataset": dataset,
                "problem_id": problem_id,
                "prompt": row.get("prompt"),
                "reference": row.get("reference"),
                "category": row.get("category"),
                "split": row.get("split"),
                "source_path": row.get("source_path"),
                "source_row_index": row.get("source_row_index"),
            }
        )
    return out


def _evaluate_mbpp_row(
    row: Dict[str, Any],
    *,
    source_cache: Dict[str, List[Dict[str, Any]]],
    timeout_sec: float,
    python_bin: str,
) -> Tuple[float, str]:
    source_path = Path(str(row.get("source_path", ""))).expanduser()
    source_row_index = int(row.get("source_row_index", 0) or 0)
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset row not found: {source_path}")
    cache_key = str(source_path.resolve())
    if cache_key not in source_cache:
        source_cache[cache_key] = _load_source_rows(source_path)
    source_rows = source_cache[cache_key]
    if source_row_index <= 0 or source_row_index > len(source_rows):
        raise IndexError(
            f"Invalid source_row_index={source_row_index} for source_path={source_path}"
        )
    source_row = source_rows[source_row_index - 1]
    tests = _extract_tests_from_source_row(source_row)
    if not tests:
        raise ValueError(f"MBPP sample {row.get('problem_id')} is missing executable tests.")
    item = MBPPItem(
        problem_id=str(row.get("problem_id", "")),
        prompt=str(row.get("prompt", "")),
        tests=tests,
        test_setup_code=str(source_row.get("test_setup_code", "") or ""),
    )
    code = _extract_code(str(row.get("prediction", "")))
    ok, err = _eval_mbpp(code, item, timeout_sec=timeout_sec, python_bin=python_bin)
    return (1.0 if ok else 0.0), str(err or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate existing Stage A code outputs without rerunning inference.")
    parser.add_argument("--root", type=Path, required=True, help="Existing Stage A directory (contains raw_predictions.jsonl).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination Stage A directory. Defaults to <root>_reeval_code_exec.",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite the existing Stage A directory in place.")
    parser.add_argument("--timeout-sec", type=float, default=4.0, help="Per-sample code execution timeout.")
    parser.add_argument("--python-bin", default="python", help="Python executable for code execution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    if args.in_place:
        output_root = root
    else:
        output_root = (args.output_root.expanduser().resolve() if args.output_root else root.parent / f"{root.name}_reeval_code_exec")
        output_root.mkdir(parents=True, exist_ok=True)

    offline_dir = output_root / "offline_supervision"
    offline_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _read_jsonl(root / "raw_predictions.jsonl")
    normalized_items = _read_jsonl(root / "train_items.normalized.jsonl")
    resolved_config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))

    source_cache: Dict[str, List[Dict[str, Any]]] = {}
    reevaluated_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        item = dict(row)
        if str(item.get("category", "")).strip().lower() == "code" and str(item.get("dataset", "")) == "mbpp":
            try:
                correctness, exec_error = _evaluate_mbpp_row(
                    item,
                    source_cache=source_cache,
                    timeout_sec=float(args.timeout_sec),
                    python_bin=str(args.python_bin),
                )
                item["is_correct"] = float(correctness)
                item["is_correct_exec"] = float(correctness)
                item["eval_status"] = "scored"
                if exec_error:
                    item["exec_error"] = exec_error
                else:
                    item.pop("exec_error", None)
                item.pop("eval_error", None)
            except Exception as exc:
                item["eval_status"] = "eval_error"
                item["eval_error"] = str(exc)
        reevaluated_rows.append(item)

    trace_input_rows: List[Dict[str, Any]] = []
    trace_skipped_rows: List[Dict[str, Any]] = []
    for row in reevaluated_rows:
        if row.get("eval_status") != "scored":
            trace_skipped_rows.append(
                {
                    "dataset": row.get("dataset"),
                    "problem_id": row.get("problem_id"),
                    "info_mode": row.get("info_mode"),
                    "cot_mode": row.get("cot_mode"),
                    "mode_tag": row.get("mode_tag"),
                    "reason": row.get("eval_status"),
                    "eval_error": row.get("eval_error"),
                }
            )
            continue
        trace_input_rows.append(row)

    trace_result = collect_static_traces(trace_input_rows, code_policy="weak_text_match", on_error="skip")

    offline_cfg = resolved_config.get("offline_supervision", {}) if isinstance(resolved_config.get("offline_supervision"), dict) else {}
    pair_cfg = offline_cfg.get("pair_generation", {}) if isinstance(offline_cfg.get("pair_generation"), dict) else {}
    hard_cfg = offline_cfg.get("hard_example", {}) if isinstance(offline_cfg.get("hard_example"), dict) else {}
    offline_summary = _prepare_offline_outputs(
        trace_result.traces,
        output_dir=offline_dir,
        lambda_token_penalty=float(offline_cfg.get("lambda_token_penalty", 0.2)),
        hidden_proj_dim=int(offline_cfg.get("hidden_proj_dim", 0) or 0),
        token_z_threshold=float(hard_cfg.get("token_z_threshold", 1.0)),
        min_score_gap=float(pair_cfg.get("min_score_gap", 0.2)),
        max_pairs_per_problem=int(pair_cfg.get("max_pairs_per_problem", 4) or 4),
        pair_strategy=str(pair_cfg.get("pair_strategy", "multi_pair") or "multi_pair"),
        disagreement_reweight=bool(pair_cfg.get("disagreement_reweight", True)),
    )

    hard_items = _hard_items_from_normalized_rows(normalized_items, offline_summary.get("hard_examples", []))

    raw_predictions_path = write_jsonl(output_root / "raw_predictions.jsonl", reevaluated_rows)
    normalized_items_path = write_jsonl(output_root / "train_items.normalized.jsonl", normalized_items)
    trace_skipped_path = write_jsonl(output_root / "trace_skipped_rows.jsonl", trace_skipped_rows)
    static_traces_path = write_jsonl(output_root / "static_traces.jsonl", trace_result.traces)
    hard_eval_items_path = write_jsonl(output_root / "hard_eval_items.jsonl", hard_items)

    if root != output_root:
        _copy_if_exists(root / "resolved_config.json", output_root / "resolved_config.json")

    summary = {
        "pipeline_config": resolved_config.get("pipeline_config"),
        "model_config": resolved_config.get("model_config"),
        "dry_run": resolved_config.get("dry_run", False),
        "stage": resolved_config.get("stage", "A"),
        "generation_seed": resolved_config.get("generation_seed"),
        "code_correctness_policy": "execute_tests_recheck",
        "re_evaluated_from": str(root),
        "output_root": str(output_root),
        "files": {
            "normalized_items": str(normalized_items_path.resolve()),
            "raw_predictions": str(raw_predictions_path.resolve()),
            "trace_skipped_rows": str(trace_skipped_path.resolve()),
            "static_traces": str(static_traces_path.resolve()),
            "hard_eval_items": str(hard_eval_items_path.resolve()),
            "resolved_config": str((output_root / "resolved_config.json").resolve()),
            **offline_summary["files"],
        },
        "counts": {
            "normalized_items": len(normalized_items),
            "raw_predictions": len(reevaluated_rows),
            "scored_raw_rows": sum(1 for row in reevaluated_rows if row.get("eval_status") == "scored"),
            "prediction_only_rows": sum(1 for row in reevaluated_rows if row.get("eval_status") == "prediction_only"),
            "eval_error_rows": sum(1 for row in reevaluated_rows if row.get("eval_status") == "eval_error"),
            "trace_skipped_rows": len(trace_skipped_rows),
            "static_traces": len(trace_result.traces),
            "trace_errors": len(trace_result.errors),
            "hard_eval_items": len(hard_items),
            **offline_summary["counts"],
        },
        "aggregate_raw": _aggregate_raw_rows(reevaluated_rows),
        "stage_a_modes": resolved_config.get("stage_a_modes", []),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
