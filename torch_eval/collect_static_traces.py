#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = PACKAGE_ROOT.parent
SRC_ROOT = METHOD_ROOT / "src"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {exc}")

from autocrat.trace_collection import collect_static_traces, summarize_traces  # noqa: E402
from autocrat_controller.io_utils import write_json  # noqa: E402
from torch_eval.eval_utils import DatasetTokenStats, _load_items, _resolve_path  # noqa: E402
from torch_eval.run_torch_eval import _load_hf_runtime, _manual_decode_problem, _set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fixed 4x5 AutoCRAT static traces with the pure PyTorch/Transformers decoder."
    )
    parser.add_argument("--config", required=True, help="YAML config with model, benchmark, and action-grid settings.")
    parser.add_argument("--output-dir", required=True, help="Directory for raw_predictions.jsonl and static_traces.jsonl.")
    parser.add_argument("--split", choices=("train", "test"), default="train", help="Divided benchmark split to collect.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset allowlist.")
    parser.add_argument("--max-items-per-dataset", type=int, default=None, help="Small smoke-test limit.")
    parser.add_argument("--max-actions", type=int, default=None, help="Optional cap on grid actions for smoke tests.")
    parser.add_argument("--device", default=None, help="Torch device when device_map is disabled, e.g. cuda:0.")
    parser.add_argument("--device-map", default=None, help="HF device_map override, e.g. auto or none.")
    return parser.parse_args()


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _cfg_for_split(cfg: Mapping[str, Any], split: str) -> Dict[str, Any]:
    out = dict(cfg)
    datasets: List[Dict[str, Any]] = []
    for spec in cfg.get("datasets", []):
        item = dict(spec)
        item["split"] = split
        item["source"] = f"{item['dataset']}/{split}.jsonl"
        item["per_dataset_limit"] = None
        datasets.append(item)
    out["datasets"] = datasets
    return out


def _action_grid(cfg: Mapping[str, Any], max_actions: int | None) -> List[Tuple[int, int]]:
    sampling = sorted(int(k) for k in cfg.get("sampling_levels", cfg.get("info_modes", {})).keys())
    budgets = sorted(int(k) for k in cfg.get("budget_levels", cfg.get("cot_budgets", {})).keys())
    actions = [(sampling_level, budget_level) for sampling_level in sampling for budget_level in budgets]
    if max_actions is not None:
        actions = actions[: max(1, int(max_actions))]
    return actions


def _dataset_token_stats() -> Dict[str, DatasetTokenStats]:
    return {"__global__": DatasetTokenStats(mean=0.0, std=1.0)}


def _raw_row_from_result(result: Any, item: Any, action: Tuple[int, int], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    info_mode, cot_mode = int(action[0]), int(action[1])
    sampling_cfg = cfg.get("sampling_levels", cfg.get("info_modes", {}))
    info_cfg = sampling_cfg[str(info_mode)] if str(info_mode) in sampling_cfg else sampling_cfg[info_mode]
    return {
        "dataset": str(item.dataset),
        "problem_id": str(item.problem_id),
        "category": str(item.category),
        "prompt": str(item.prompt),
        "reference": str(item.reference),
        "info_mode": info_mode,
        "cot_mode": cot_mode,
        "prediction": str(result.prediction),
        "is_correct": float(result.correctness),
        "token_count": int(result.token_count),
        "think_zone_token_count": int(result.think_token_count),
        "answer_zone_token_count": int(result.answer_token_count),
        "finish_reason": str(result.finish_reason),
        "temperature_actual": float(info_cfg.get("temperature", 0.0)),
        "top_p_actual": float(info_cfg.get("top_p", 0.95)),
        "top_k_actual": int(info_cfg.get("top_k", -1)),
        "boundary_count": len(result.meta.get("boundary_states", [])),
        "boundary_states": list(result.meta.get("boundary_states", [])),
        "metadata": {
            "runtime": "torch_manual_kv_cache",
            "split": str(item.split),
            "source_path": str(item.source_path),
            "source_row_index": int(item.source_row_index),
            "eval_error": str(result.meta.get("error", "")),
        },
    }


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = _cfg_for_split(cfg, str(args.split))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items_args = argparse.Namespace(datasets=args.datasets, max_items_per_dataset=args.max_items_per_dataset)
    items_by_dataset = _load_items(cfg_path, cfg, items_args)
    actions = _action_grid(cfg, args.max_actions)
    torch_mod, model, tokenizer = _load_hf_runtime(cfg_path, cfg, args)
    _set_seed(torch_mod, cfg)

    runtime_cfg = cfg["policy_runtime"]
    runtime = cfg["runtime"]
    model_cfg = cfg["model"]
    prompt_template = str(cfg["prompt_template"])
    info_modes = cfg.get("sampling_levels", cfg.get("info_modes"))
    cot_budgets = cfg.get("budget_levels", cfg.get("cot_budgets"))
    timeout_sec = float(runtime.get("code_exec_timeout_sec", 4.0))
    python_bin = str(runtime.get("code_exec_python_bin", "python3"))

    raw_rows: List[Dict[str, Any]] = []
    for dataset, items in items_by_dataset.items():
        for item in items:
            for action in actions:
                result = _manual_decode_problem(
                    torch_mod=torch_mod,
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    prompt_template=prompt_template,
                    policy=None,
                    info_modes=info_modes,
                    cot_budgets=cot_budgets,
                    start_action=action,
                    baseline_name="static_grid",
                    allow_switches=False,
                    runtime_cfg=runtime_cfg,
                    model_cfg=model_cfg,
                    timeout_sec=timeout_sec,
                    python_bin=python_bin,
                    lambda_penalty=0.0,
                    dataset_token_stats=_dataset_token_stats(),
                )
                raw_rows.append(_raw_row_from_result(result, item, action, cfg))
        print(f"[static_trace] dataset={dataset} items={len(items)} actions={len(actions)}", flush=True)

    collection = collect_static_traces(raw_rows, code_policy="require_label", on_error="raise")
    _write_jsonl(output_dir / "raw_predictions.jsonl", raw_rows)
    _write_jsonl(output_dir / "static_traces.jsonl", collection.traces)
    summary = {
        "runtime": "torch_manual_kv_cache",
        "split": str(args.split),
        "actions": [{"info_mode": int(i), "cot_mode": int(c)} for i, c in actions],
        "datasets": {dataset: len(items) for dataset, items in items_by_dataset.items()},
        "raw_predictions": len(raw_rows),
        "static_traces": len(collection.traces),
        "trace_summary": summarize_traces(collection.traces),
        "errors": collection.errors,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
