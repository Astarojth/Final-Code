from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .settings import COT_MODE_TABLE, DATASET_REGISTRY, INFO_MODE_TABLE, validate_mode_tables


def _resolve_dataset_specs(dataset_keys: List[str]) -> List[Dict[str, Any]]:
    resolved = []
    for key in dataset_keys:
        if key not in DATASET_REGISTRY:
            raise KeyError(f"Unknown dataset key: {key}")
        resolved.append(asdict(DATASET_REGISTRY[key]))
    return resolved


def build_bootstrap_plan(experiment_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    validate_mode_tables()
    selected = experiment_cfg.get("datasets", {}).get("selected", [])
    if not isinstance(selected, list) or not selected:
        raise ValueError("`datasets.selected` must be a non-empty list.")
    dataset_specs = _resolve_dataset_specs(selected)

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": {
            "name": experiment_cfg.get("experiment_name", "autocrat_bootstrap"),
            "description": experiment_cfg.get("description", ""),
            "lambda_token_penalty": experiment_cfg.get("lambda_token_penalty", 0.2),
            "seed": experiment_cfg.get("seed", 42),
        },
        "model": model_cfg,
        "datasets": dataset_specs,
        "info_modes": [asdict(INFO_MODE_TABLE[k]) for k in sorted(INFO_MODE_TABLE)],
        "cot_modes": [asdict(COT_MODE_TABLE[k]) for k in sorted(COT_MODE_TABLE)],
        "static_mode_candidates": experiment_cfg.get("static_mode_candidates", []),
    }
    return plan


def write_plan(plan: Dict[str, Any], runs_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = runs_root / f"bootstrap_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "plan.json"
    out_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path
