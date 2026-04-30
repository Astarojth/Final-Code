#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autocrat.config import load_yaml, resolve_path
from autocrat.offline_supervision import (
    build_boundary_preference_pairs,
    build_boundary_samples,
    build_preference_pairs,
    find_hard_examples,
    read_jsonl,
    score_traces,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare offline supervision artifacts from static traces.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/offline_supervision.yaml"),
        help="Path to offline supervision config.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override input traces jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory.",
    )
    return parser.parse_args()


def _resolve_overrides(cfg_path: Path, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(cfg)
    if args.input:
        merged["input_traces"] = args.input
    if args.output_dir:
        merged["output_dir"] = args.output_dir

    if not merged.get("input_traces"):
        raise ValueError("Config must set `input_traces` or pass --input.")
    if not merged.get("output_dir"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged["output_dir"] = f"runs/offline_supervision_{ts}"

    merged["input_traces"] = str(resolve_path(cfg_path, str(merged["input_traces"])))
    merged["output_dir"] = str(resolve_path(cfg_path, str(merged["output_dir"])))
    return merged


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    cfg = _resolve_overrides(cfg_path, cfg, args)

    lambda_token = float(cfg.get("lambda_token_penalty", 0.2))
    supervision_mode = str(cfg.get("supervision_mode", "boundary")).strip().lower() or "boundary"
    hard_cfg = cfg.get("hard_example", {})
    pair_cfg = cfg.get("pair_generation", {})

    traces = read_jsonl(cfg["input_traces"])
    scored = score_traces(traces, lambda_token_penalty=lambda_token)
    hard_examples = find_hard_examples(
        scored,
        token_z_threshold=float(hard_cfg.get("token_z_threshold", 1.0)),
    )
    pairs = build_preference_pairs(
        scored,
        min_score_gap=float(pair_cfg.get("min_score_gap", 0.2)),
        max_pairs_per_problem=int(pair_cfg.get("max_pairs_per_problem", 4)),
        pair_strategy=str(pair_cfg.get("pair_strategy", "multi_pair")),
        disagreement_reweight=bool(pair_cfg.get("disagreement_reweight", True)),
    )
    boundary_samples = build_boundary_samples(scored)
    boundary_pairs = build_boundary_preference_pairs(
        boundary_samples,
        min_score_gap=float(pair_cfg.get("min_score_gap", 0.2)),
        max_pairs_per_problem=int(pair_cfg.get("max_pairs_per_problem", 4)),
        pair_strategy=str(pair_cfg.get("pair_strategy", "multi_pair")),
        disagreement_reweight=bool(pair_cfg.get("disagreement_reweight", True)),
    )

    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = write_jsonl(out_dir / "scored_traces.jsonl", scored)
    hard_path = write_jsonl(out_dir / "hard_examples.jsonl", hard_examples)
    pairs_path = write_jsonl(out_dir / "preference_pairs.jsonl", pairs)
    boundary_samples_path = write_jsonl(out_dir / "boundary_samples.jsonl", boundary_samples)
    boundary_pairs_path = write_jsonl(out_dir / "boundary_preference_pairs.jsonl", boundary_pairs)

    summary = {
        "input_traces": str(Path(cfg["input_traces"]).resolve()),
        "output_dir": str(out_dir),
        "lambda_token_penalty": lambda_token,
        "supervision_mode": supervision_mode,
        "counts": {
            "traces": len(traces),
            "scored_traces": len(scored),
            "hard_examples": len(hard_examples),
            "preference_pairs": len(pairs),
            "boundary_samples": len(boundary_samples),
            "boundary_preference_pairs": len(boundary_pairs),
        },
        "files": {
            "scored_traces": str(scored_path),
            "hard_examples": str(hard_path),
            "preference_pairs": str(pairs_path),
            "boundary_samples": str(boundary_samples_path),
            "boundary_preference_pairs": str(boundary_pairs_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "boundary_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
