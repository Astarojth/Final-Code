from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_fixture_supervision_trains_final_controller(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture = repo_root / "tests" / "fixtures" / "static_traces.sample.jsonl"
    stage_root = tmp_path / "stage_a"
    offline_dir = stage_root / "offline_supervision"
    controller_dir = tmp_path / "controller"

    prepare = subprocess.run(
        [
            "python",
            str(repo_root / "scripts" / "prepare_offline_supervision.py"),
            "--config",
            str(repo_root / "configs" / "offline_supervision.yaml"),
            "--input",
            str(fixture),
            "--output-dir",
            str(offline_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stderr

    train = subprocess.run(
        [
            "python",
            str(repo_root / "autocrat_controller" / "train_controller.py"),
            "--supervision-dir",
            str(stage_root),
            "--output-dir",
            str(controller_dir),
            "--text-dim",
            "32",
            "--hidden-dim",
            "16",
            "--num-slots",
            "4",
            "--prior-epochs",
            "1",
            "--think-boundary-epochs",
            "1",
            "--answer-boundary-epochs",
            "1",
            "--disable-val",
            "--num-workers",
            "1",
            "--boundary-buffer-max-mb",
            "128",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert train.returncode == 0, train.stderr

    summary = json.loads((controller_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["scored_traces"] == 12
    assert summary["counts"]["boundary_samples"] > 0
    assert summary["controller_training_epochs"] == 3
    assert (controller_dir / "autocrat_controller.pt").exists()
