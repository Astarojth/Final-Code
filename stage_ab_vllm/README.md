# Stage A/B vLLM Pipeline

This folder contains the local-data Stage A pipeline for AutoCRAT's 4x5 static-policy trace collection.

## Files

- `stage_ab_vllm_pipeline.py`
  - Loads local benchmark files
  - Normalizes them into canonical Stage A items
  - Runs static vLLM collection for the configured Stage A mode set
  - Writes raw predictions, canonical traces, scored traces, hard examples, and boundary-level artifacts
- `configs/model.qwen3_4b_thinking_2507.vllm.yaml`
  - Model/backend config for `Qwen/Qwen3-4B-Thinking-2507`
  - Supports `download_dir` / `cache_dir` for persistent local Hugging Face caching
- `configs/stage_a.qwen3_4b_thinking.local.yaml`
  - End-to-end Stage A config using `/path/to/benchmark_usable`

## Usage

```bash
python3 stage_ab_vllm/stage_ab_vllm_pipeline.py \
  --config stage_ab_vllm/configs/stage_a.qwen3_4b_thinking.local.yaml
```

For a quick smoke run:

```bash
python3 stage_ab_vllm/stage_ab_vllm_pipeline.py \
  --config stage_ab_vllm/configs/stage_a.qwen3_4b_thinking.local.yaml \
  --max-items-per-dataset 5
```

## Outputs

The default output root is:

`runs/stage_ab_vllm/qwen3_4b_thinking_2507_stage_a/stage_a/`

Important files:

- `train_items.normalized.jsonl`
- `raw_predictions.jsonl`
- `trace_skipped_rows.jsonl`
- `static_traces.jsonl`
- `offline_supervision/scored_traces.jsonl`
- `offline_supervision/hard_examples.jsonl`
- `offline_supervision/preference_pairs.jsonl`
- `offline_supervision/boundary_samples.jsonl`
- `offline_supervision/boundary_preference_pairs.jsonl`
- `hard_eval_items.jsonl`
- `summary.json`
