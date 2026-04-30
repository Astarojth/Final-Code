from __future__ import annotations

import json
import subprocess
from pathlib import Path

from autocrat.governance_model import load_governance_bundle
from autocrat.mock_training import (
    DualHeadTrainingConfig,
    FeatureEncoder,
    build_bc_dataset,
    build_preference_dataset,
    train_dual_head_model,
)
from autocrat.offline_supervision import (
    build_boundary_preference_pairs,
    build_boundary_samples,
    build_preference_pairs,
    read_jsonl,
    score_traces,
    write_jsonl,
)


def _raw_fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "static_traces.sample.jsonl"


def _prepare_scored_and_pairs(tmp_path: Path) -> tuple[Path, Path]:
    raw = read_jsonl(_raw_fixture_path())
    scored = score_traces(raw, lambda_token_penalty=0.2)
    pairs = build_preference_pairs(scored, min_score_gap=0.2, max_pairs_per_problem=4)
    scored_path = write_jsonl(tmp_path / "scored.jsonl", scored)
    pairs_path = write_jsonl(tmp_path / "pairs.jsonl", pairs)
    return scored_path, pairs_path


def test_build_datasets_shapes(tmp_path: Path) -> None:
    scored_path, pairs_path = _prepare_scored_and_pairs(tmp_path)
    scored = read_jsonl(scored_path)
    pairs = read_jsonl(pairs_path)

    encoder = FeatureEncoder.fit(scored)
    x_bc, y_info, y_cot = build_bc_dataset(scored, encoder)
    pref = build_preference_dataset(scored, pairs, encoder)

    assert x_bc.shape[0] == 4
    assert x_bc.shape[1] == encoder.feature_dim
    assert y_info.shape[0] == x_bc.shape[0]
    assert y_cot.shape[0] == x_bc.shape[0]
    assert pref.x_pos.shape == pref.x_neg.shape
    assert pref.x_pos.shape[1] == encoder.feature_dim
    assert pref.sample_weights.shape[0] == pref.x_pos.shape[0]


def test_build_boundary_datasets_shapes(tmp_path: Path) -> None:
    scored_path, _ = _prepare_scored_and_pairs(tmp_path)
    scored = read_jsonl(scored_path)
    boundary_samples = build_boundary_samples(scored, hidden_proj_dim=8)
    boundary_pairs = build_boundary_preference_pairs(boundary_samples, min_score_gap=0.2, max_pairs_per_problem=2)
    encoder = FeatureEncoder.fit(boundary_samples, hidden_proj_dim=8)
    x_bc, y_info, y_cot = build_bc_dataset(boundary_samples, encoder, supervision_mode="boundary")
    pref = build_preference_dataset(boundary_samples, boundary_pairs, encoder, supervision_mode="boundary")

    assert x_bc.shape[0] == len(boundary_samples)
    assert x_bc.shape[1] == encoder.feature_dim
    assert pref.x_pos.shape[1] == encoder.feature_dim


def test_boundary_dataset_requires_full_governance_context(tmp_path: Path) -> None:
    scored_path, _ = _prepare_scored_and_pairs(tmp_path)
    scored = read_jsonl(scored_path)
    boundary_samples = build_boundary_samples(scored, hidden_proj_dim=8)
    assert boundary_samples

    bad = dict(boundary_samples[0])
    bad_state = dict(bad["state_features"])
    bad_state.pop("current_cot_mode", None)
    bad["state_features"] = bad_state

    encoder = FeatureEncoder.fit(boundary_samples, hidden_proj_dim=8)
    try:
        _ = build_bc_dataset([bad], encoder, supervision_mode="boundary")
        raise AssertionError("Expected ValueError for missing governance context field.")
    except ValueError as exc:
        assert "current_cot_mode" in str(exc)


def test_dual_head_trainer_returns_valid_metrics(tmp_path: Path) -> None:
    scored_path, pairs_path = _prepare_scored_and_pairs(tmp_path)
    scored = read_jsonl(scored_path)
    pairs = read_jsonl(pairs_path)

    encoder = FeatureEncoder.fit(scored)
    x_bc, y_info, y_cot = build_bc_dataset(scored, encoder)
    pref = build_preference_dataset(scored, pairs, encoder)

    result = train_dual_head_model(
        x_bc,
        y_info,
        y_cot,
        pref,
        cfg=DualHeadTrainingConfig(hidden_dim=12, steps=40, lr=0.05, seed=7),
    )

    metrics = result["metrics"]
    assert 0.0 <= metrics["bc_info_acc"] <= 1.0
    assert 0.0 <= metrics["bc_cot_acc"] <= 1.0
    assert 0.0 <= metrics["pair_acc"] <= 1.0
    assert result["model"].input_dim == encoder.feature_dim


def test_train_script_end_to_end(tmp_path: Path) -> None:
    scored_path, pairs_path = _prepare_scored_and_pairs(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_mock_governance.py"
    output_dir = tmp_path / "mock_out"
    scored = read_jsonl(scored_path)
    boundary_samples = build_boundary_samples(scored, hidden_proj_dim=8)
    boundary_pairs = build_boundary_preference_pairs(boundary_samples, min_score_gap=0.2, max_pairs_per_problem=2)
    boundary_samples_path = write_jsonl(tmp_path / "boundary_samples.jsonl", boundary_samples)
    boundary_pairs_path = write_jsonl(tmp_path / "boundary_pairs.jsonl", boundary_pairs)
    cfg = {
        "scored_traces": str(scored_path),
        "preference_pairs": str(pairs_path),
        "boundary_samples": str(boundary_samples_path),
        "boundary_preference_pairs": str(boundary_pairs_path),
        "output_dir": str(output_dir),
        "seed": 7,
        "hidden_proj_dim": 8,
        "supervision_mode": "boundary",
        "model": {"hidden_dim": 8},
        "bc": {"steps": 8, "lr": 0.05, "l2": 1e-4},
        "preference": {"weight": 0.3},
        "task_encoder": {"enabled": True, "dim": 24, "slot_schema_version": "v12_15slot"},
        "memory": {
            "type": "slot_soft_matching",
            "target_slot_count": 15,
            "slot_merge_cosine_threshold": 0.9,
            "slot_diversity_weight": 0.02,
            "slot_diversity_margin": 0.35,
            "slot_match_temperature_init": 1.0,
            "slot_match_temperature_learnable": True,
            "slot_schema_version": "v12_15slot",
        },
    }
    cfg_path = tmp_path / "mock_training.fast.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            str(script),
            "--config",
            str(cfg_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["counts"]["scored_traces"] == 12
    assert metrics["counts"]["preference_pairs"] > 0
    assert metrics["counts"]["boundary_samples"] > 0
    assert metrics["supervision_mode"] == "boundary"
    assert metrics["final_slot_count"] == 15
    assert metrics["slot_schema_version"] == "v12_15slot"
    assert len(metrics["final_slot_names"]) == 15
    assert "slot_merge_report" in metrics
    assert "match_temperature" in metrics
    assert "diversity_loss_stats" in metrics
    model_path = output_dir / "model.json"
    assert model_path.exists()

    bundle = load_governance_bundle(model_path)
    assert bundle["encoder"].feature_dim == bundle["model"].input_dim
    assert bundle["metadata"]["hidden_dim"] > 0
    assert bundle["metadata"]["feature_schema_version"] == "boundary_v2"
    assert bundle["metadata"]["slot_schema_version"] == "v12_15slot"
    assert len(bundle["metadata"]["final_slot_names"]) == 15
