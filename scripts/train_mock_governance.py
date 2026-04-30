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
from autocrat.adaptive_governance import SlotMemory, TaskEncoder
from autocrat.mock_training import (
    FeatureEncoder,
    build_bc_dataset,
    build_preference_dataset,
    save_training_artifact,
    train_dual_head_model,
)
from autocrat.offline_supervision import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-head governance model from offline traces.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "mock_training.yaml"),
        help="Path to training config.",
    )
    parser.add_argument("--scored-traces", default=None, help="Override scored traces path.")
    parser.add_argument("--preference-pairs", default=None, help="Override preference pairs path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser.parse_args()


def _resolve(cfg_path: Path, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(cfg)
    if args.scored_traces:
        merged["scored_traces"] = args.scored_traces
    if args.preference_pairs:
        merged["preference_pairs"] = args.preference_pairs
    if args.output_dir:
        merged["output_dir"] = args.output_dir

    for key in ("scored_traces", "preference_pairs"):
        if not merged.get(key):
            raise ValueError(f"Config must set `{key}` or pass corresponding CLI arg.")
    if not merged.get("output_dir"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged["output_dir"] = f"runs/mock_training_{ts}"

    merged["scored_traces"] = str(resolve_path(cfg_path, str(merged["scored_traces"])))
    merged["preference_pairs"] = str(resolve_path(cfg_path, str(merged["preference_pairs"])))
    if merged.get("boundary_samples"):
        merged["boundary_samples"] = str(resolve_path(cfg_path, str(merged["boundary_samples"])))
    if merged.get("boundary_preference_pairs"):
        merged["boundary_preference_pairs"] = str(resolve_path(cfg_path, str(merged["boundary_preference_pairs"])))
    merged["output_dir"] = str(resolve_path(cfg_path, str(merged["output_dir"])))
    return merged


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _resolve(cfg_path, load_yaml(cfg_path), args)

    seed = int(cfg.get("seed", 42))
    supervision_mode = str(cfg.get("supervision_mode", "boundary")).strip().lower() or "boundary"
    hidden_proj_dim = int(cfg.get("hidden_proj_dim", 0) or 0)
    use_hidden_observables = bool(cfg.get("use_hidden_observables", True))
    feature_schema_version = str(cfg.get("feature_schema_version", "boundary_v2") or "boundary_v2")
    train_split_manifest = str(cfg.get("train_split_manifest", "") or "")
    online_memory_update_enabled = bool(cfg.get("online_memory_update_enabled", True))
    bc_cfg = cfg.get("bc", {}) if isinstance(cfg.get("bc"), dict) else {}
    pref_cfg = cfg.get("preference", {}) if isinstance(cfg.get("preference"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    task_encoder_cfg = cfg.get("task_encoder", {}) if isinstance(cfg.get("task_encoder"), dict) else {}
    memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}

    scored_traces = read_jsonl(cfg["scored_traces"])
    preference_pairs = read_jsonl(cfg["preference_pairs"])
    boundary_samples = read_jsonl(cfg["boundary_samples"]) if cfg.get("boundary_samples") else []
    boundary_pairs = read_jsonl(cfg["boundary_preference_pairs"]) if cfg.get("boundary_preference_pairs") else []
    train_rows = boundary_samples if supervision_mode == "boundary" and boundary_samples else scored_traces
    train_pairs = boundary_pairs if supervision_mode == "boundary" and boundary_pairs else preference_pairs
    encoder = FeatureEncoder.fit(
        train_rows,
        hidden_proj_dim=hidden_proj_dim,
        feature_schema_version=feature_schema_version,
        use_hidden_observables=use_hidden_observables,
    )

    task_encoder_enabled = bool(task_encoder_cfg.get("enabled", False))
    task_encoder_dim = int(task_encoder_cfg.get("dim", 24) or 24)
    slot_schema_version = str(memory_cfg.get("slot_schema_version", task_encoder_cfg.get("slot_schema_version", "v12_15slot")))
    task_encoder = TaskEncoder(dim=task_encoder_dim, slot_schema_version=slot_schema_version)
    slot_memory = SlotMemory(
        encoder_dim=task_encoder_dim,
        target_slot_count=int(memory_cfg.get("target_slot_count", 15) or 15),
        merge_cosine_threshold=float(memory_cfg.get("slot_merge_cosine_threshold", 0.90) or 0.90),
        diversity_weight=float(memory_cfg.get("slot_diversity_weight", 0.02) or 0.02),
        diversity_margin=float(memory_cfg.get("slot_diversity_margin", 0.35) or 0.35),
        match_temperature_init=float(memory_cfg.get("slot_match_temperature_init", 1.0) or 1.0),
        match_temperature_learnable=bool(memory_cfg.get("slot_match_temperature_learnable", True)),
        slot_schema_version=slot_schema_version,
    )
    slot_rows_used = 0
    if task_encoder_enabled:
        slot_rows_used = int(slot_memory.initialize_from_rows(task_encoder=task_encoder, rows=train_rows))
    slot_diversity_loss = float(slot_memory.diversity_regularization_loss())
    slot_merge_report = slot_memory.last_merge_report
    slot_diversity_loss_stats = slot_memory.last_diversity_loss_stats
    slot_temperature = float(slot_memory.match_temperature)
    final_slot_names = slot_memory.final_slot_names

    x_bc, y_info, y_cot = build_bc_dataset(train_rows, encoder, supervision_mode=supervision_mode)
    preference_data = build_preference_dataset(train_rows, train_pairs, encoder, supervision_mode=supervision_mode)

    result = train_dual_head_model(
        x_bc,
        y_info,
        y_cot,
        preference_data,
        hidden_dim=int(model_cfg.get("hidden_dim", 16)),
        steps=int(bc_cfg.get("steps", 400)),
        lr=float(bc_cfg.get("lr", 0.03)),
        l2=float(bc_cfg.get("l2", 1e-4)),
        preference_weight=float(pref_cfg.get("weight", 0.5)),
        slot_diversity_weight=float(memory_cfg.get("slot_diversity_weight", 0.02) or 0.02),
        slot_diversity_loss=float(slot_diversity_loss if task_encoder_enabled else 0.0),
        seed=seed,
    )

    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.json"
    save_training_artifact(
        str(model_path),
        encoder=encoder,
        model=result["model"],
        metadata={
            "seed": seed,
            "hidden_dim": int(model_cfg.get("hidden_dim", 16)),
            "hidden_proj_dim": hidden_proj_dim,
            "uses_hidden": bool(use_hidden_observables),
            "online_update": bool(online_memory_update_enabled),
            "steps": int(bc_cfg.get("steps", 400)),
            "lr": float(bc_cfg.get("lr", 0.03)),
            "weight_decay": float(bc_cfg.get("l2", 1e-4)),
            "preference_weight": float(pref_cfg.get("weight", 0.5)),
            "feature_schema_version": encoder.feature_schema_version,
            "supervision_mode": supervision_mode,
            "train_split_manifest": train_split_manifest,
            "task_encoder": task_encoder.to_dict(),
            "slot_memory": slot_memory.to_dict() if task_encoder_enabled else {},
            "memory_type": str(memory_cfg.get("type", "legacy_action_ema")),
            "slot_schema_version": str(slot_memory.slot_schema_version),
            "final_slot_names": final_slot_names,
            "slot_merge_report": slot_merge_report,
            "match_temperature": slot_temperature,
            "diversity_loss_stats": slot_diversity_loss_stats,
        },
    )

    metrics = {
        "counts": {
            "scored_traces": len(scored_traces),
            "preference_pairs": len(preference_pairs),
            "boundary_samples": len(boundary_samples),
            "boundary_preference_pairs": len(boundary_pairs),
            "bc_samples": int(x_bc.shape[0]),
            "rank_pairs": int(preference_data.x_pos.shape[0]),
        },
        "feature_dim": encoder.feature_dim,
        "feature_schema_version": encoder.feature_schema_version,
        "use_hidden_observables": bool(use_hidden_observables),
        "online_memory_update_enabled": bool(online_memory_update_enabled),
        "task_encoder_enabled": bool(task_encoder_enabled),
        "slot_rows_used": int(slot_rows_used),
        "slot_schema_version": str(slot_memory.slot_schema_version),
        "final_slot_count": int(len(final_slot_names)),
        "final_slot_names": final_slot_names,
        "slot_merge_report": slot_merge_report,
        "match_temperature": float(slot_temperature),
        "diversity_loss_stats": slot_diversity_loss_stats,
        "supervision_mode": supervision_mode,
        "metrics": result["metrics"],
        "artifacts": {"model": str(model_path)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "config.resolved.json").write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **metrics}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
