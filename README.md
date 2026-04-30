# AutoCRAT Final Code

This repository contains the cleaned source code for the AutoCRAT final project. It is a compact reproducibility package for the paper's final method: a decoder-side controller that jointly updates sampling stochasticity and reasoning budget at semantic boundaries.

Datasets, model weights, trained controller checkpoints, generated runs, logs, and result bundles are intentionally not included.

## What Is Included

- `policy20_training/`: 4x5 controller action space, boundary features, MLP control heads, policy logic, Stage-A replay, and controller training.
- `stage_ab_vllm/`: Stage-A static trace collection and offline-supervision artifact construction with vLLM.
- `policy20_torch_eval/`: manual PyTorch online evaluator that keeps the KV cache alive and applies boundary-level controller updates.
- `src/autocrat/`: shared grading, trace collection, offline supervision, config, and vLLM post-processing utilities.
- `scripts/`: benchmark split construction, offline-supervision prep, lightweight governance training, and code execution scoring.
- `tests/`: small fixtures and lightweight regression tests. These fixtures are synthetic examples, not benchmark data.

## Paper-Aligned Control Interface

AutoCRAT uses a discrete 4x5 control grid:

- Sampling levels: temperatures `0.0`, `0.3`, `0.7`, `1.0`.
- Budget levels: thinking-token budgets `0`, `64`, `256`, `1024`, `4096`.
- Boundary updates: controller decisions are made at semantic boundary points, with the default minimum boundary interval set to `6` generated tokens.

The controller training defaults match the appendix settings: hidden dimension `256`, dropout `0.1`, AdamW, learning rate `3e-4`, weight decay `1e-4`, target softmax temperature `0.25`, and pairwise loss weight `0.5`. The implementation trains phase-specific heads with `40 + 100 + 100` epochs, which is the code-level equivalent of the paper's 140-epoch controller training schedule.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For real model runs, install the backend dependencies needed by your runtime:

```bash
pip install -r requirements.custom-backend.txt
```

vLLM is required for the Stage-A vLLM pipeline. Install a version compatible with your CUDA and PyTorch environment.

## Recreate Local Benchmark Splits

The repository does not include public benchmark data. To build local split files from public Hugging Face datasets:

```bash
python scripts/build_eval_items_benchmarks.py \
  --output /path/to/benchmark_usable/eval_items.benchmarks_tiered_v5.jsonl
```

For the paper experiments, configure the six benchmark sources locally:

- GSM8K
- MATH-500
- ARC-Challenge
- GPQA-Diamond
- HumanEval
- MBPP

Use `/path/to/benchmark_usable` as the `dataset_root` placeholder in the included configs.

## Pipeline

1. Collect Stage-A traces over the 4x5 static grid:

```bash
python stage_ab_vllm/stage_ab_vllm_pipeline.py \
  --config stage_ab_vllm/configs/stage_a.qwen3_4b_thinking.local.yaml
```

2. Train the boundary-level controller:

```bash
python policy20_training/train_stagea20.py \
  --stage-a-dir /path/to/stage_a \
  --output-dir /path/to/policy20_train_out \
  --prior-epochs 40 \
  --think-boundary-epochs 100 \
  --answer-boundary-epochs 100
```

3. Run online evaluation with the trained controller:

```bash
python policy20_torch_eval/torch_online_eval.py \
  --config policy20_torch_eval/config.qwen3_4b_thinking_2507.torch.local.yaml \
  --output-dir /path/to/policy20_torch_eval \
  --baselines segmented_global_best_no_switch dynamic_global_best_start
```

4. Score code-generation outputs with execution tests when needed:

```bash
python scripts/eval_code_exec.py --root /path/to/eval_outputs --output-dir /path/to/code_exec_eval
```

## Local Paths To Fill In

Before running large experiments, edit the YAML configs and replace:

- `/path/to/benchmark_usable`
- `/path/to/stage_a`
- `/path/to/policy20_training.pt`
- `/path/to/model_cache`

Do not commit the resolved local paths if they point to private data, model caches, or experiment outputs.

## Tests

```bash
pytest -q tests/test_offline_supervision.py tests/test_vllm_post_process.py tests/test_mock_training.py tests/test_settings.py
```

Import smoke check:

```bash
PYTHONPATH=src:. python - <<'PY'
import autocrat
import policy20_training
import policy20_torch_eval
import stage_ab_vllm
print("ok")
PY
```
