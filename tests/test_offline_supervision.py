from __future__ import annotations

import json
import subprocess
from pathlib import Path

from autocrat.offline_supervision import (
    build_boundary_preference_pairs,
    build_boundary_samples,
    build_preference_pairs,
    find_hard_examples,
    read_jsonl,
    score_traces,
)


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "static_traces.sample.jsonl"


def test_score_traces_enriches_records() -> None:
    traces = read_jsonl(_fixture_path())
    scored = score_traces(traces, lambda_token_penalty=0.2)
    assert len(scored) == len(traces)
    assert all("token_z" in row and "score" in row for row in scored)


def test_hard_examples_flags_all_incorrect_problem() -> None:
    traces = read_jsonl(_fixture_path())
    scored = score_traces(traces, lambda_token_penalty=0.2)
    hard = find_hard_examples(scored, token_z_threshold=1.0)
    gsm8k_p2 = next(item for item in hard if item["problem_id"] == "gsm8k_p2")
    assert "all_incorrect" in gsm8k_p2["reasons"]


def test_preference_pairs_respect_min_gap() -> None:
    traces = read_jsonl(_fixture_path())
    scored = score_traces(traces, lambda_token_penalty=0.2)
    pairs = build_preference_pairs(scored, min_score_gap=0.2, max_pairs_per_problem=4)
    assert pairs
    assert all(pair["score_gap"] >= 0.2 for pair in pairs)
    assert all(float(pair.get("pair_weight", 0.0)) > 0.0 for pair in pairs)
    assert any(
        pair["winner_trace_id"] == "gsm8k_p1_i3_c2" and pair["loser_trace_id"] == "gsm8k_p1_i2_c3"
        for pair in pairs
    )


def test_boundary_samples_and_pairs_are_generated() -> None:
    traces = read_jsonl(_fixture_path())
    scored = score_traces(traces, lambda_token_penalty=0.2)
    boundary_samples = build_boundary_samples(scored, hidden_proj_dim=8)
    assert boundary_samples
    assert all("state_features" in row for row in boundary_samples)
    assert all(len(row["state_features"]["hidden_state_proj"]) == 8 for row in boundary_samples)

    boundary_pairs = build_boundary_preference_pairs(boundary_samples, min_score_gap=0.2, max_pairs_per_problem=2)
    assert boundary_pairs
    assert all(pair["winner_boundary_id"].endswith(f"#{pair['boundary_index']}") for pair in boundary_pairs)


def test_boundary_samples_prefer_boundary_states_payload() -> None:
    scored = [
        {
            "trace_id": "gsm8k_x_i3_c2",
            "dataset": "gsm8k",
            "problem_id": "gsm8k_x",
            "category": "math",
            "prompt": "2+2=?",
            "score": 1.0,
            "token_count": 12,
            "info_mode": 3,
            "cot_mode": 2,
            "boundary_states": [
                {
                    "entropy": 0.4,
                    "margin": 0.2,
                    "top1_prob": 0.6,
                    "top2_prob": 0.3,
                    "topk_mass": [0.6, 0.9, 0.95, 0.97, 0.98],
                    "topk_mass_5": 0.98,
                    "eos_prob": 0.01,
                    "eos_rank": 9.0,
                    "repeat_ngram_ratio": 0.0,
                    "generated_tokens": 4,
                    "progress_ratio": 0.25,
                    "current_info_mode": 3,
                    "current_cot_mode": 2,
                    "remaining_budget_ratio": 0.75,
                    "segment_progress_ratio": 0.5,
                    "is_answer_zone": False,
                    "is_code_mode": False,
                    "boundary_kind": "punct",
                    "chosen_info_mode": 3,
                    "chosen_cot_mode": 2,
                }
            ],
        }
    ]
    boundary_samples = build_boundary_samples(scored, hidden_proj_dim=0)
    assert len(boundary_samples) == 1
    assert boundary_samples[0]["state_features"]["topk_mass"] == 0.98
    assert boundary_samples[0]["boundary_kind"] == "punct"


def test_prepare_script_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "prepare_offline_supervision.py"
    output_dir = tmp_path / "offline_out"

    result = subprocess.run(
        [
            "python",
            str(script),
            "--config",
            str(repo_root / "configs" / "offline_supervision.yaml"),
            "--input",
            str(_fixture_path()),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["traces"] == 12
    assert (output_dir / "scored_traces.jsonl").exists()
    assert (output_dir / "hard_examples.jsonl").exists()
    assert (output_dir / "preference_pairs.jsonl").exists()
    assert (output_dir / "boundary_samples.jsonl").exists()
    assert (output_dir / "boundary_preference_pairs.jsonl").exists()
