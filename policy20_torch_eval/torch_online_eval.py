#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

from policy20_torch_eval.online_eval_compat import (  # noqa: E402
    FINAL_ANSWER_MARKER,
    EvalItem,
    _best_static_action,
    _best_static_actions_by_category,
    _build_boundary_row,
    _category_key,
    _compute_dataset_token_stats,
    _cot_budget,
    _evaluate_prediction,
    _infer_lambda,
    _load_action_space,
    _load_boundary_spec,
    _load_items,
    _load_jsonl,
    _load_model,
    _load_slot_memory,
    _repeat_ngram_ratio,
    _resolve_path,
    _resolve_start_action,
    _score_proxy,
    _torch_load_artifact,
)
from policy20_training.features import HashTextVectorizer  # noqa: E402
from policy20_training.policy import OnlineDecisionPolicy  # noqa: E402
from policy20_training.io_utils import write_json  # noqa: E402


SUPPORTED_BASELINES = {
    "segmented_global_best_no_switch",
    "segmented_category_best_no_switch",
    "dynamic_global_best_start",
    "dynamic_category_best_start",
    "torch_global_best_no_switch",
    "torch_category_best_no_switch",
    "torch_dynamic_global_best_start",
    "torch_dynamic_category_best_start",
}

BASELINE_ALIASES = {
    "torch_global_best_no_switch": "segmented_global_best_no_switch",
    "torch_category_best_no_switch": "segmented_category_best_no_switch",
    "torch_dynamic_global_best_start": "dynamic_global_best_start",
    "torch_dynamic_category_best_start": "dynamic_category_best_start",
}


@dataclass(frozen=True)
class TorchProblemResult:
    baseline: str
    dataset: str
    problem_id: str
    correctness: float
    token_count: int
    think_token_count: int
    answer_token_count: int
    latency_sec: float
    switches: int
    start_action: Tuple[int, int]
    final_action: Tuple[int, int]
    prediction: str
    score_proxy: float
    finish_reason: str
    meta: Dict[str, Any]


@dataclass
class StepFeatures:
    entropy: float = 0.0
    margin: float = 0.0
    top1_prob: float = 0.0
    top2_prob: float = 0.0
    topk_mass: float = 0.0
    eos_prob: float = 0.0
    eos_rank: float = 0.0


def _boundary_kind(decoded_tail: str) -> str:
    if not decoded_tail:
        return ""
    if "\n" in decoded_tail:
        return "newline"
    stripped = decoded_tail.rstrip()
    if stripped.endswith((".", "?", "!", "。", "？", "！", ":", "：")):
        return "punct"
    low = stripped.lower()
    if "step " in low and ":" in low[-24:]:
        return "step_marker"
    if "reasoning:" in low[-24:]:
        return "step_marker"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="True PyTorch/KV-cache online eval for policy20.")
    parser.add_argument("--config", required=True, help="YAML config path for policy20_torch_eval.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--max-items-per-dataset", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset allowlist.")
    parser.add_argument("--baselines", nargs="*", default=None, help="Optional baseline allowlist.")
    parser.add_argument("--device", default=None, help="Torch device when device_map is disabled, e.g. cuda:0.")
    parser.add_argument("--device-map", default=None, help="HF device_map override, e.g. auto or none.")
    return parser.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def _canonical_baseline(name: str) -> str:
    return BASELINE_ALIASES.get(str(name), str(name))


def _ordered_baselines(names: Sequence[str]) -> List[str]:
    priority = {
        "segmented_global_best_no_switch": 0,
        "segmented_category_best_no_switch": 1,
        "dynamic_global_best_start": 2,
        "dynamic_category_best_start": 3,
    }
    canonical = [_canonical_baseline(name) for name in names]
    seen: set[str] = set()
    out: List[str] = []
    for name in canonical:
        if name in SUPPORTED_BASELINES and name not in seen:
            seen.add(name)
            out.append(name)
    return sorted(out, key=lambda item: (priority.get(item, 99), item))


def _summarize(rows: Sequence[TorchProblemResult]) -> Dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "acc": 0.0,
            "score": 0.0,
            "tokens": 0.0,
            "think_tokens": 0.0,
            "answer_tokens": 0.0,
            "latency_sec": 0.0,
            "switches": 0.0,
        }
    return {
        "n": float(len(rows)),
        "acc": sum(row.correctness for row in rows) / len(rows),
        "score": sum(row.score_proxy for row in rows) / len(rows),
        "tokens": sum(row.token_count for row in rows) / len(rows),
        "think_tokens": sum(row.think_token_count for row in rows) / len(rows),
        "answer_tokens": sum(row.answer_token_count for row in rows) / len(rows),
        "latency_sec": sum(row.latency_sec for row in rows) / len(rows),
        "switches": sum(row.switches for row in rows) / len(rows),
    }


def _print_progress(baseline: str, completed: int, total: int, rows: Sequence[TorchProblemResult]) -> None:
    summary = _summarize(rows)
    _log(
        f"[{baseline}] {completed}/{total} "
        f"acc={summary['acc']:.3f} score={summary['score']:.3f} "
        f"avg_tokens={summary['tokens']:.1f} "
        f"think={summary['think_tokens']:.1f} answer={summary['answer_tokens']:.1f} "
        f"avg_switches={summary['switches']:.2f}"
    )


def _load_policy(artifact_path: Path, policy_runtime: Mapping[str, Any]) -> OnlineDecisionPolicy:
    artifact = _torch_load_artifact(artifact_path)
    action_space = _load_action_space(artifact)
    text_vectorizer = HashTextVectorizer(**artifact["text_vectorizer"])
    slot_memory = _load_slot_memory(artifact)
    boundary_spec = _load_boundary_spec(artifact)
    prior_model = _load_model(artifact, "prior_model")
    think_boundary_model = _load_model(artifact, "think_boundary_model")
    answer_boundary_model = _load_model(artifact, "answer_boundary_model")
    return OnlineDecisionPolicy(
        action_space=action_space,
        text_vectorizer=text_vectorizer,
        boundary_spec=boundary_spec,
        slot_memory=slot_memory,
        prior_model=prior_model,
        think_boundary_model=think_boundary_model,
        answer_boundary_model=answer_boundary_model,
        prior_weight=float(policy_runtime["prior_weight"]),
        boundary_weight=float(policy_runtime["boundary_weight"]),
        switch_cost=float(policy_runtime["switch_cost"]),
        hysteresis_bonus=float(policy_runtime["hysteresis_bonus"]),
        budget_guardrail_penalty=float(policy_runtime["budget_guardrail_penalty"]),
    )


def _dtype_from_config(torch_mod: Any, dtype_name: str) -> Any:
    name = str(dtype_name or "bfloat16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch_mod.float16
    if name in {"fp32", "float32", "float"}:
        return torch_mod.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _first_model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return getattr(model, "device", "cpu")


def _load_hf_runtime(cfg_path: Path, cfg: Mapping[str, Any], args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg["model"]
    download_dir = str(model_cfg.get("download_dir", "")).strip() or None
    cache_dir = str(model_cfg.get("cache_dir", download_dir or "")).strip() or None
    hf_home = str(model_cfg.get("hf_home", "")).strip() or None
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))

    torch_cfg = cfg.get("torch_runtime", {}) or {}
    device_map_value = args.device_map
    if device_map_value is None:
        device_map_value = torch_cfg.get("device_map", "auto")
    if str(device_map_value).lower() in {"none", "null", "false", ""}:
        device_map_value = None

    dtype = _dtype_from_config(torch, str(model_cfg.get("dtype", "bfloat16")))
    common = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "cache_dir": cache_dir,
    }
    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg["tokenizer_dir"]), **common)
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {
        **common,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": bool(torch_cfg.get("low_cpu_mem_usage", True)),
    }
    if device_map_value is not None:
        model_kwargs["device_map"] = device_map_value
    model = AutoModelForCausalLM.from_pretrained(str(model_cfg["model_dir"]), **model_kwargs)
    if device_map_value is None:
        device = args.device or torch_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    _log(
        "Loaded HF runtime: "
        f"model={model_cfg['model_dir']} tokenizer={model_cfg['tokenizer_dir']} "
        f"device={_first_model_device(model)} device_map={device_map_value} cache_dir={cache_dir}"
    )
    return torch, model, tokenizer


def _set_seed(torch_mod: Any, cfg: Mapping[str, Any]) -> None:
    torch_cfg = cfg.get("torch_runtime", {}) or {}
    seed = torch_cfg.get("seed", cfg.get("generation_seed", cfg.get("sampling_seed", 20260409)))
    seed = int(seed)
    random.seed(seed)
    torch_mod.manual_seed(seed)
    if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)
    _log(f"Torch sampling seed: {seed}")


def _render_prompt(prompt_template: str, item: EvalItem) -> str:
    return prompt_template.format(prompt=item.prompt)


def _chat_template_enabled(tokenizer: Any, model_cfg: Mapping[str, Any]) -> bool:
    mode = str(model_cfg.get("chat_template_mode", "auto")).strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    has_template = hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer, "chat_template", None))
    if mode == "on":
        return bool(has_template)
    if mode == "off":
        return False
    return bool(has_template)


def _render_model_prompt(tokenizer: Any, model_cfg: Mapping[str, Any], prompt: str) -> str:
    if not _chat_template_enabled(tokenizer, model_cfg):
        return prompt
    messages: List[Dict[str, str]] = []
    system_prompt = str(model_cfg.get("chat_system_prompt", "")).strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=bool(model_cfg.get("chat_template_add_generation_prompt", True)),
            )
        )
    except Exception:
        return prompt


def _as_token_id_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        return [int(x) for x in value if x is not None]
    return []


def _sample_next_token(
    torch_mod: Any,
    logits: Any,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> Any:
    if float(temperature) <= 1e-8:
        return torch_mod.argmax(logits, dim=-1, keepdim=True)

    scores = logits / float(temperature)
    if int(top_k) > 0:
        values, _ = torch_mod.topk(scores, k=min(int(top_k), scores.shape[-1]), dim=-1)
        threshold = values[..., -1, None]
        scores = torch_mod.where(scores < threshold, torch_mod.full_like(scores, -float("inf")), scores)

    probs = torch_mod.softmax(scores, dim=-1)
    if float(min_p) > 0.0:
        max_prob = probs.max(dim=-1, keepdim=True).values
        probs = torch_mod.where(probs < max_prob * float(min_p), torch_mod.zeros_like(probs), probs)

    if 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch_mod.sort(probs, descending=True, dim=-1)
        cumulative = torch_mod.cumsum(sorted_probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch_mod.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

    denom = probs.sum(dim=-1, keepdim=True)
    if bool((denom <= 0).any()):
        return torch_mod.argmax(logits, dim=-1, keepdim=True)
    probs = probs / denom
    return torch_mod.multinomial(probs, num_samples=1)


def _logit_features(torch_mod: Any, logits: Any, *, eos_token_ids: Sequence[int], top_k: int) -> StepFeatures:
    with torch_mod.no_grad():
        logprobs = torch_mod.log_softmax(logits, dim=-1)
        probs = torch_mod.exp(logprobs)
        top_n = min(max(2, int(top_k or 50)), logits.shape[-1])
        top_probs, top_indices = torch_mod.topk(probs, k=top_n, dim=-1)
        top_logprobs = torch_mod.gather(logprobs, -1, top_indices)
        entropy = -torch_mod.sum(probs * logprobs, dim=-1)
        top1_prob = top_probs[:, 0]
        top2_prob = top_probs[:, 1] if top_probs.shape[-1] > 1 else torch_mod.zeros_like(top1_prob)
        margin = top_logprobs[:, 0] - (top_logprobs[:, 1] if top_logprobs.shape[-1] > 1 else top_logprobs[:, 0])
        topk_mass = torch_mod.sum(top_probs[:, : min(5, top_probs.shape[-1])], dim=-1)
        eos_prob = torch_mod.zeros_like(top1_prob)
        eos_rank = torch_mod.full_like(top1_prob, float(top_n + 1))
        for eos_id in eos_token_ids:
            matches = top_indices == int(eos_id)
            if bool(matches.any()):
                rank_tensor = torch_mod.argmax(matches.to(torch_mod.int64), dim=-1)
                has_match = matches.any(dim=-1)
                eos_rank = torch_mod.where(has_match, rank_tensor.to(eos_rank.dtype) + 1.0, eos_rank)
                eos_prob = torch_mod.where(has_match, top_probs.gather(-1, rank_tensor[:, None]).squeeze(-1), eos_prob)
        return StepFeatures(
            entropy=float(entropy[0].detach().cpu()),
            margin=float(margin[0].detach().cpu()),
            top1_prob=float(top1_prob[0].detach().cpu()),
            top2_prob=float(top2_prob[0].detach().cpu()),
            topk_mass=float(topk_mass[0].detach().cpu()),
            eos_prob=float(eos_prob[0].detach().cpu()),
            eos_rank=float(eos_rank[0].detach().cpu()),
        )


class ManualPolicyController:
    def __init__(
        self,
        *,
        item: EvalItem,
        policy: Optional[OnlineDecisionPolicy],
        start_action: Tuple[int, int],
        info_modes: Mapping[str, Any],
        cot_budgets: Mapping[str, Any],
        runtime_cfg: Mapping[str, Any],
        tokenizer: Any,
        reasoning_end_str: str,
        allow_switches: bool,
    ) -> None:
        self.item = item
        self.policy = policy
        self.current_action = (int(start_action[0]), int(start_action[1]))
        self.start_action = self.current_action
        self.info_modes = info_modes
        self.cot_budgets = cot_budgets
        self.runtime_cfg = runtime_cfg
        self.tokenizer = tokenizer
        self.allow_switches = bool(allow_switches)
        self.reasoning_end_str = str(reasoning_end_str)
        self.close_str = "</think>"
        suffix = self.reasoning_end_str
        if suffix.startswith(self.close_str):
            suffix = suffix[len(self.close_str) :]
        self.final_suffix_after_close = suffix
        self.reasoning_end_ids = [
            int(x) for x in tokenizer.encode(self.reasoning_end_str, add_special_tokens=False)
        ]
        self.close_token_ids = [
            int(x) for x in tokenizer.encode(self.close_str, add_special_tokens=False)
        ]
        self.final_suffix_ids = [
            int(x) for x in tokenizer.encode(self.final_suffix_after_close, add_special_tokens=False)
        ]
        self.forced_queue: List[int] = []
        self.close_bias_active = False
        self.in_answer_zone = False
        self.think_used = 0
        self.answer_used = 0
        self.token_count = 0
        self.switches = 0
        self.boundary_index = 0
        self.tokens_since_boundary = 0
        self.boundary_count = 0
        self.decoded_tail = ""
        self.generated_token_ids: List[int] = []
        self.last_features = StepFeatures()

    def cot_budget(self) -> int:
        return _cot_budget(self.cot_budgets, int(self.current_action[1]))

    def info_cfg(self) -> Mapping[str, Any]:
        info = int(self.current_action[0])
        return self.info_modes[str(info)] if str(info) in self.info_modes else self.info_modes[info]

    def boundary_min_tokens(self) -> int:
        return max(1, int(self.runtime_cfg.get("boundary_min_tokens", 6) or 6))

    def max_boundaries(self) -> int:
        return max(1, int(self.runtime_cfg.get("max_segments", 80)))

    def queue_reasoning_end_if_needed(self) -> None:
        if self.in_answer_zone or self.forced_queue:
            return
        if self.think_used >= self.cot_budget():
            control = str(self.runtime_cfg.get("reasoning_end_control", "force")).strip().lower()
            if control == "bias_close" and self.close_token_ids:
                self.close_bias_active = True
            else:
                self.forced_queue.extend(self.reasoning_end_ids)

    def sync_answer_zone_from_text(self, generated_text: str) -> None:
        if self.in_answer_zone:
            return
        close_idx = generated_text.lower().rfind(self.close_str)
        if close_idx < 0:
            return
        self.in_answer_zone = True
        self.close_bias_active = False
        tail = generated_text[close_idx + len(self.close_str) :]
        if FINAL_ANSWER_MARKER not in tail and not self.forced_queue:
            self.forced_queue.extend(self.final_suffix_ids)

    def choose_next_token(self, torch_mod: Any, logits: Any) -> Tuple[Any, bool]:
        self.queue_reasoning_end_if_needed()
        if self.forced_queue:
            token_id = int(self.forced_queue.pop(0))
            return torch_mod.tensor([[token_id]], dtype=torch_mod.long, device=logits.device), True
        if self.close_bias_active and self.close_token_ids:
            biased = logits.clone()
            bias = float(self.runtime_cfg.get("reasoning_end_token_bias", 100.0))
            biased[:, int(self.close_token_ids[0])] += bias
            logits = biased
        cfg = self.info_cfg()
        token = _sample_next_token(
            torch_mod,
            logits,
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            top_k=int(cfg.get("top_k", -1)),
            min_p=float(cfg.get("min_p", 0.0)),
        )
        return token, False

    def record_token(self, token_id: int, features: StepFeatures, generated_text: str) -> None:
        self.last_features = features
        self.generated_token_ids.append(int(token_id))
        self.token_count += 1
        self.tokens_since_boundary += 1
        piece = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        self.decoded_tail = (self.decoded_tail + str(piece))[-128:]
        if self.in_answer_zone:
            self.answer_used += 1
        else:
            self.think_used += 1
        self.sync_answer_zone_from_text(generated_text)

    def is_boundary(self, *, eos_reached: bool) -> Tuple[bool, str]:
        if eos_reached:
            return True, "eos"
        kind = _boundary_kind(self.decoded_tail)
        if self.tokens_since_boundary >= self.boundary_min_tokens() and kind:
            return True, kind
        return False, ""

    def maybe_decide(self, *, eos_reached: bool) -> None:
        if self.forced_queue:
            return
        is_boundary, boundary_kind = self.is_boundary(eos_reached=eos_reached)
        if not is_boundary:
            return
        self.boundary_count += 1
        if self.boundary_count > self.max_boundaries():
            return
        self.tokens_since_boundary = 0
        self.decoded_tail = ""
        if not self.allow_switches:
            return
        if self.policy is None:
            raise RuntimeError("Dynamic torch baseline requires a loaded policy.")
        features = self.last_features
        if self.in_answer_zone and boundary_kind not in {"eos", "newline", "punct", "step_marker"}:
            boundary_kind = "answer_ready"
        boundary_row = _build_boundary_row(
            item=self.item,
            generated_tokens=self.token_count,
            think_budget=self.cot_budget(),
            think_used=self.think_used,
            in_answer_zone=self.in_answer_zone,
            boundary_kind=boundary_kind or ("answer_ready" if self.in_answer_zone else "segment_budget"),
            entropy=features.entropy,
            margin=features.margin,
            top1_prob=features.top1_prob,
            top2_prob=features.top2_prob,
            topk_mass=features.topk_mass,
            eos_prob=features.eos_prob,
            eos_rank=features.eos_rank,
            repeat_ngram_ratio=_repeat_ngram_ratio(self.generated_token_ids, n=3, window=128),
        )
        boundary_row["boundary_index"] = int(self.boundary_index)
        self.boundary_index += 1
        remaining = max(0, self.cot_budget() - self.think_used)
        decision = self.policy.choose_boundary_action(
            prompt=self.item.prompt,
            boundary_row=boundary_row,
            current_action=self.current_action,
            remaining_thinking_budget_tokens=remaining,
        )
        next_action = (
            int(decision["best_action"]["info_mode"]),
            int(decision["best_action"]["cot_mode"]),
        )
        if (
            next_action != self.current_action
            and self.switches < int(self.runtime_cfg.get("max_switches", 8))
        ):
            self.current_action = next_action
            self.switches += 1


def _manual_decode_problem(
    *,
    torch_mod: Any,
    model: Any,
    tokenizer: Any,
    item: EvalItem,
    prompt_template: str,
    policy: Optional[OnlineDecisionPolicy],
    info_modes: Mapping[str, Any],
    cot_budgets: Mapping[str, Any],
    start_action: Tuple[int, int],
    baseline_name: str,
    allow_switches: bool,
    runtime_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    timeout_sec: float,
    python_bin: str,
    lambda_penalty: float,
    dataset_token_stats: Mapping[str, Any],
) -> TorchProblemResult:
    rendered_prompt = _render_model_prompt(tokenizer, model_cfg, _render_prompt(prompt_template, item))
    device = _first_model_device(model)
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    reasoning_end_str = str(model_cfg.get("reasoning_end_str", "</think>\nFINAL_ANSWER: "))
    controller = ManualPolicyController(
        item=item,
        policy=policy,
        start_action=start_action,
        info_modes=info_modes,
        cot_budgets=cot_budgets,
        runtime_cfg=runtime_cfg,
        tokenizer=tokenizer,
        reasoning_end_str=reasoning_end_str,
        allow_switches=allow_switches,
    )

    eos_token_ids = _as_token_id_list(getattr(tokenizer, "eos_token_id", None))
    max_cot = max(int(v) for v in cot_budgets.values()) if cot_budgets else 4096
    default_answer_budget = int(runtime_cfg.get("static_answer_completion_budget", 16384) or 16384)
    prompt_tokens = int(input_ids.shape[-1])
    model_len = int(model_cfg.get("max_model_len", 32768) or 32768)
    default_max_total = max(1, min(max_cot + default_answer_budget, max(1, model_len - prompt_tokens)))
    max_total_tokens = int(runtime_cfg.get("torch_max_total_tokens", default_max_total) or default_max_total)
    logprobs_k = int(runtime_cfg.get("logprobs_k", 50))

    generated_ids: List[int] = []
    generated_text = ""
    finish_reason = "length"
    past_key_values = None
    start_time = time.perf_counter()

    with torch_mod.no_grad():
        for _ in range(max_total_tokens):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            features = _logit_features(torch_mod, logits, eos_token_ids=eos_token_ids, top_k=logprobs_k)
            next_token, _forced = controller.choose_next_token(torch_mod, logits)
            next_id = int(next_token[0, 0].detach().cpu())
            generated_ids.append(next_id)
            eos_reached = next_id in eos_token_ids
            if eos_reached:
                finish_reason = "eos"
                break
            piece = tokenizer.decode([next_id], skip_special_tokens=False)
            generated_text += str(piece)
            controller.record_token(next_id, features, generated_text)

            if controller.boundary_count >= controller.max_boundaries() and not controller.forced_queue:
                finish_reason = "max_segments"
                break

            input_ids = next_token.to(device)
            if attention_mask is not None:
                ones = torch_mod.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch_mod.cat([attention_mask, ones], dim=1)
            controller.maybe_decide(eos_reached=eos_reached)

    latency = time.perf_counter() - start_time
    correctness, err = _evaluate_prediction(item, generated_text, timeout_sec=timeout_sec, python_bin=python_bin)
    score_proxy = _score_proxy(
        dataset=item.dataset,
        correctness=float(correctness),
        token_count=len(generated_ids),
        lambda_penalty=lambda_penalty,
        dataset_token_stats=dataset_token_stats,
    )
    return TorchProblemResult(
        baseline=baseline_name,
        dataset=item.dataset,
        problem_id=item.problem_id,
        correctness=float(correctness),
        token_count=len(generated_ids),
        think_token_count=int(controller.think_used),
        answer_token_count=int(controller.answer_used),
        latency_sec=float(latency),
        switches=int(controller.switches),
        start_action=(int(start_action[0]), int(start_action[1])),
        final_action=(int(controller.current_action[0]), int(controller.current_action[1])),
        prediction=generated_text,
        score_proxy=float(score_proxy),
        finish_reason=finish_reason,
        meta={"error": err, "allow_switches": bool(allow_switches)},
    )


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _resolve_path(cfg_path, str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_baselines = list(args.baselines or cfg.get("baselines", []))
    skipped = [name for name in requested_baselines if _canonical_baseline(name) not in SUPPORTED_BASELINES]
    baselines = _ordered_baselines(requested_baselines)
    if skipped:
        _log(f"Skipping non-torch baselines: {', '.join(skipped)}")
    if not baselines:
        raise SystemExit("No supported torch baselines selected.")

    stage_a_dir = _resolve_path(cfg_path, str(cfg["stage_a_dir"]))
    artifact_path = _resolve_path(cfg_path, str(cfg["policy_artifact"]))
    items_by_dataset = _load_items(cfg_path, cfg, args)
    stage_a_scored = _load_jsonl(stage_a_dir / "offline_supervision" / "scored_traces.jsonl")
    lambda_penalty = _infer_lambda(stage_a_scored)
    dataset_token_stats = _compute_dataset_token_stats(stage_a_scored)
    best_static_action = _best_static_action(stage_a_dir)
    category_best_static_actions = _best_static_actions_by_category(stage_a_dir)
    _log(
        "Best starts: "
        f"global=({best_static_action[0]}, {best_static_action[1]}) "
        f"by_category={{{', '.join(f'{k}: ({v[0]}, {v[1]})' for k, v in sorted(category_best_static_actions.items()))}}}"
    )

    policy_runtime = cfg["policy_runtime"]
    needs_policy = any(name in {"dynamic_global_best_start", "dynamic_category_best_start"} for name in baselines)
    policy = _load_policy(artifact_path, policy_runtime) if needs_policy else None

    torch_mod, model, tokenizer = _load_hf_runtime(cfg_path, cfg, args)
    _set_seed(torch_mod, cfg)
    runtime = cfg["runtime"]
    prompt_template = str(cfg["prompt_template"])
    timeout_sec = float(runtime.get("code_exec_timeout_sec", 4.0))
    python_bin = str(runtime.get("code_exec_python_bin", "python3"))
    log_every = int(runtime.get("progress_log_every", 5))
    info_modes = cfg["info_modes"]
    cot_budgets = cfg["cot_budgets"]
    model_cfg = cfg["model"]

    all_results: Dict[str, List[TorchProblemResult]] = {}
    dataset_results: Dict[str, Dict[str, List[TorchProblemResult]]] = defaultdict(dict)
    for baseline in baselines:
        _log(f"Running torch baseline={baseline} ...")
        baseline_results: List[TorchProblemResult] = []
        allow_switches = baseline in {"dynamic_global_best_start", "dynamic_category_best_start"}
        start_strategy = (
            "category_best_static"
            if baseline in {"segmented_category_best_no_switch", "dynamic_category_best_start"}
            else "global_best_static"
        )
        for dataset, items in items_by_dataset.items():
            dataset_rows: List[TorchProblemResult] = []
            for idx, item in enumerate(items, start=1):
                start_action = _resolve_start_action(
                    strategy=start_strategy,
                    prompt=item.prompt,
                    policy=policy,  # type: ignore[arg-type]
                    action_space=getattr(policy, "action_space", None),  # type: ignore[arg-type]
                    best_static_action=best_static_action,
                    category=item.category,
                    category_best_static_actions=category_best_static_actions,
                )
                result = _manual_decode_problem(
                    torch_mod=torch_mod,
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    prompt_template=prompt_template,
                    policy=policy,
                    info_modes=info_modes,
                    cot_budgets=cot_budgets,
                    start_action=start_action,
                    baseline_name=baseline,
                    allow_switches=allow_switches,
                    runtime_cfg=policy_runtime,
                    model_cfg=model_cfg,
                    timeout_sec=timeout_sec,
                    python_bin=python_bin,
                    lambda_penalty=lambda_penalty,
                    dataset_token_stats=dataset_token_stats,
                )
                baseline_results.append(result)
                dataset_rows.append(result)
                if idx % max(1, log_every) == 0 or idx == len(items):
                    _print_progress(f"{baseline}:{dataset}", idx, len(items), dataset_rows)
            dataset_results[baseline][dataset] = dataset_rows
            summary = _summarize(dataset_rows)
            _log(
                f"[dataset_done] baseline={baseline} dataset={dataset} "
                f"n={int(summary['n'])} acc={summary['acc']:.3f} score={summary['score']:.3f} "
                f"tokens={summary['tokens']:.1f} think={summary['think_tokens']:.1f} "
                f"answer={summary['answer_tokens']:.1f} switches={summary['switches']:.2f}"
            )
        all_results[baseline] = baseline_results

    summary = {
        "runtime": "torch_manual_kv_cache",
        "best_static_action": {"info_mode": int(best_static_action[0]), "cot_mode": int(best_static_action[1])},
        "category_best_static_actions": {
            category: {"info_mode": int(action[0]), "cot_mode": int(action[1])}
            for category, action in sorted(category_best_static_actions.items())
        },
        "score_metric": "correctness - lambda_token_penalty * dataset_token_z_proxy",
        "lambda_token_penalty": float(lambda_penalty),
        "datasets": {dataset: len(items) for dataset, items in items_by_dataset.items()},
        "metrics": {baseline: _summarize(rows) for baseline, rows in all_results.items()},
    }
    write_json(output_dir / "torch_online_eval_summary.json", summary)
    for baseline, rows in all_results.items():
        payload = [asdict(row) for row in rows]
        (output_dir / f"{baseline}.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=True) for item in payload) + ("\n" if payload else ""),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
