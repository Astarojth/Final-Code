from __future__ import annotations

import math
import os
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol, Sequence

from .governance_model import compress_hidden_state
from .settings import COT_MODE_TABLE, INFO_MODE_TABLE, cot_thinking_token_budget


@dataclass(frozen=True)
class GenerationResult:
    prediction: str
    token_count: int
    raw: Dict[str, Any]


OBSERVABLE_KEYS: tuple[str, ...] = (
    "top1_prob",
    "top2_prob",
    "topk_mass",
    "eos_prob",
    "eos_rank",
    "repeat_ngram_ratio",
)


def _repeat_ngram_ratio(token_ids: Sequence[int], *, n: int = 3, window: int = 128) -> float:
    ids = list(int(x) for x in token_ids[-max(1, int(window)) :])
    if len(ids) < n:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for i in range(0, len(ids) - n + 1):
        total += 1
        ng = tuple(ids[i : i + n])
        if ng in seen:
            repeats += 1
        seen.add(ng)
    if total <= 0:
        return 0.0
    return float(repeats) / float(total)


def _aggregate_series(values: Sequence[float]) -> Dict[str, float]:
    xs = [float(x) for x in values]
    if not xs:
        return {"mean": 0.0, "last": 0.0, "slope": 0.0}
    last = float(xs[-1])
    mean = float(sum(xs) / max(1, len(xs)))
    if len(xs) <= 1:
        slope = 0.0
    else:
        slope = float(xs[-1] - xs[0]) / float(len(xs) - 1)
    return {"mean": mean, "last": last, "slope": slope}


def _decode_prediction_with_forced_trigger(
    tokenizer: Any,
    *,
    generated_ids: Sequence[int],
    answer_trigger_text: str,
    forced_trigger_insert_at: int | None,
) -> str:
    ids = [int(x) for x in generated_ids]
    trigger = str(answer_trigger_text)
    if forced_trigger_insert_at is None or not trigger.strip():
        return str(tokenizer.decode(ids, skip_special_tokens=True))
    insert_at = max(0, min(int(forced_trigger_insert_at), len(ids)))
    prefix = str(tokenizer.decode(ids[:insert_at], skip_special_tokens=True))
    suffix = str(tokenizer.decode(ids[insert_at:], skip_special_tokens=True))
    return f"{prefix}{trigger}{suffix}"


class CustomBackend(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        ...

    def generate_batch(
        self,
        prompts: List[str],
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_texts: Sequence[str] | None = None,
        answer_completion_budget: int = 0,
    ) -> List[GenerationResult]:
        ...

    def probe_prompts(self, prompts: List[str]) -> List[Dict[str, Any]]:
        ...

    def generate_dynamic(
        self,
        prompt: str,
        *,
        initial_info_mode: int,
        initial_cot_mode: int,
        on_boundary: Callable[[Dict[str, Any]], tuple[int, int, Dict[str, Any]]],
        max_new_tokens: int | None = None,
        min_new_tokens_before_eos: int = 0,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        ...


class DryRunCustomBackend:
    """Deterministic backend for pipeline validation without model inference."""

    def __init__(self, model_cfg: Dict[str, Any] | None = None) -> None:
        self.hidden_state_proj_dim = int((model_cfg or {}).get("hidden_state_proj_dim", 0) or 0)

    def generate(
        self,
        prompt: str,
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        cot = COT_MODE_TABLE[cot_mode]
        head = prompt.strip().splitlines()[0] if prompt.strip() else "EMPTY_PROMPT"
        prediction = f"[DRYRUN i{info_mode} c{cot_mode}] {head[:120]}"
        if int(answer_completion_budget) > 0:
            if str(answer_trigger_text).strip():
                prediction += " FINAL_ANSWER: dry_run"
            else:
                prediction += " answer_complete"
        token_count = int(
            max(
                8,
                12
                + cot_thinking_token_budget(
                    cot,
                    max_new_tokens_per_step=self.max_new_tokens_per_step if hasattr(self, "max_new_tokens_per_step") else 16,
                )
                // 8
                + math.ceil(len(prompt) / 64),
            )
        )
        return GenerationResult(
            prediction=prediction,
            token_count=token_count,
            raw={"backend": "dry_run"},
        )

    def generate_batch(
        self,
        prompts: List[str],
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_texts: Sequence[str] | None = None,
        answer_completion_budget: int = 0,
    ) -> List[GenerationResult]:
        if answer_trigger_texts is None:
            answer_trigger_texts = [""] * len(prompts)
        return [
            self.generate(
                prompt,
                info_mode=info_mode,
                cot_mode=cot_mode,
                answer_trigger_text=str(trigger),
                answer_completion_budget=answer_completion_budget,
            )
            for prompt, trigger in zip(prompts, answer_trigger_texts)
        ]

    def probe_prompts(self, prompts: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for prompt in prompts:
            p = str(prompt or "")
            prompt_len = max(1, len(p))
            out.append(
                {
                    "entropy": 5.0 + min(3.0, (prompt_len / 600.0)),
                    "margin": 1.0 / max(1.0, math.log1p(prompt_len)),
                    "top1_token_id": 0,
                    "top1_logit": 0.0,
                    "top1_prob": 0.65,
                    "top2_prob": 0.20,
                    "topk_mass": 0.95,
                    "eos_prob": 0.01,
                    "eos_rank": 99.0,
                    "repeat_ngram_ratio": 0.0,
                    "hidden_l2_norm": 0.0,
                    "hidden_state_proj": [0.0] * max(0, int(self.hidden_state_proj_dim)),
                }
            )
        return out

    def generate_dynamic(
        self,
        prompt: str,
        *,
        initial_info_mode: int,
        initial_cot_mode: int,
        on_boundary: Callable[[Dict[str, Any]], tuple[int, int, Dict[str, Any]]],
        max_new_tokens: int | None = None,
        min_new_tokens_before_eos: int = 0,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        cot = COT_MODE_TABLE[int(initial_cot_mode)]
        thinking_budget = int(
            max_new_tokens
            or max(
                12,
                cot_thinking_token_budget(
                    cot,
                    max_new_tokens_per_step=16,
                ),
            )
        )
        info_mode = int(initial_info_mode)
        cot_mode = int(initial_cot_mode)
        pieces: List[str] = []
        mode_trace: List[Dict[str, Any]] = []
        segment_total = max(8, thinking_budget // 2)
        segment_obs: Dict[str, List[float]] = {k: [] for k in OBSERVABLE_KEYS}
        in_answer_phase = False
        answer_budget_remaining = max(0, int(answer_completion_budget))
        step = 0
        while True:
            if not in_answer_phase and step >= max(2, thinking_budget // 8):
                if answer_budget_remaining <= 0:
                    break
                in_answer_phase = True
            if in_answer_phase and answer_budget_remaining <= 0:
                break
            token_piece = f" step{step + 1}."
            pieces.append(token_piece)
            segment_obs["top1_prob"].append(max(0.05, 0.72 - (0.02 * step)))
            segment_obs["top2_prob"].append(min(0.40, 0.15 + (0.01 * step)))
            segment_obs["topk_mass"].append(0.95)
            segment_obs["eos_prob"].append(min(0.8, 0.01 * float(step + 1)))
            segment_obs["eos_rank"].append(max(1.0, 120.0 - float(step * 3)))
            segment_obs["repeat_ngram_ratio"].append(0.0 if step < 3 else min(0.5, step / 20.0))
            if (step + 1) % 2 == 0:
                segment_progress = min(1.0, float((step + 1) * 4) / float(max(1, segment_total)))
                agg = {k: _aggregate_series(vs) for k, vs in segment_obs.items()}
                boundary_state = {
                    "entropy": 5.0 + min(2.0, len(prompt) / 900.0) + (0.05 * step),
                    "margin": max(0.05, 0.9 - (0.03 * step)),
                    "top1_prob": agg["top1_prob"]["last"],
                    "top2_prob": agg["top2_prob"]["last"],
                    "topk_mass": agg["topk_mass"]["last"],
                    "eos_prob": agg["eos_prob"]["last"],
                    "eos_rank": agg["eos_rank"]["last"],
                    "repeat_ngram_ratio": agg["repeat_ngram_ratio"]["last"],
                    "boundary_observables": agg,
                    "prompt_len": len(prompt) + len("".join(pieces)),
                    "generated_tokens": (step + 1) * 4,
                    "current_info_mode": info_mode,
                    "current_cot_mode": cot_mode,
                    "is_boundary": True,
                    "decoded_tail": "".join(pieces)[-96:],
                    "eos_reached": False,
                    "boundary_kind": "segment_budget",
                    "segment_progress_ratio": segment_progress,
                    "hidden_state_proj": [0.0] * max(0, int(self.hidden_state_proj_dim)),
                }
                new_info, new_cot, extra = on_boundary(boundary_state)
                info_mode, cot_mode = int(new_info), int(new_cot)
                mode_trace.append(
                    {
                        "token_index": (step + 1) * 4,
                        "info_mode": info_mode,
                        "cot_mode": cot_mode,
                        "entropy": boundary_state["entropy"],
                        "margin": boundary_state["margin"],
                        "boundary_kind": boundary_state["boundary_kind"],
                        "segment_progress_ratio": segment_progress,
                        "hidden_state_proj": list(boundary_state["hidden_state_proj"]),
                        "extra": extra,
                    }
                )
                segment_obs = {k: [] for k in OBSERVABLE_KEYS}
                if cot_mode == 0 and not in_answer_phase:
                    in_answer_phase = True
                elif cot_mode == 0:
                    break
            if in_answer_phase:
                answer_budget_remaining -= 1
            step += 1
        prediction = f"[DRYRUN-DYNAMIC i{initial_info_mode} c{initial_cot_mode}] {prompt[:120]} {''.join(pieces)}".strip()
        if in_answer_phase and str(answer_trigger_text).strip():
            prediction += " FINAL_ANSWER: dry_run"
        token_count = min(
            thinking_budget + max(0, int(answer_completion_budget)),
            max(8, len("".join(pieces).split()) * 3),
        )
        return GenerationResult(
            prediction=prediction,
            token_count=token_count,
            raw={
                "backend": "dry_run",
                "dynamic": True,
                "mode_trace": mode_trace,
                "final_info_mode": info_mode,
                "final_cot_mode": cot_mode,
                "answer_completion_budget": int(answer_completion_budget),
                "in_answer_phase": bool(in_answer_phase),
            },
        )


class TransformersCustomBackend:
    """Custom decode loop with local transformers backend (no API dependency)."""

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Transformers backend requires `torch` and `transformers`. "
                "Install backend deps before running non-dry baseline."
            ) from exc

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_dir = str(model_cfg["model_dir"])
        self.tokenizer_dir = str(model_cfg.get("tokenizer_dir", self.model_dir))
        self.download_dir = str(model_cfg.get("download_dir", "")).strip() or None
        self.cache_dir = str(model_cfg.get("cache_dir", self.download_dir or "")).strip() or None
        hf_home = str(model_cfg.get("hf_home", "")).strip()
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
            os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))
        self.trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        self.max_new_tokens_per_step = int(model_cfg.get("max_new_tokens_per_step", 16))
        self.input_max_length = int(model_cfg.get("input_max_length", 0))
        self.dynamic_boundary_min_tokens = int(model_cfg.get("dynamic_boundary_min_tokens", 6))
        self.dynamic_min_new_tokens_no_eos = int(model_cfg.get("dynamic_min_new_tokens_no_eos", 0))
        self.hidden_state_proj_dim = int(model_cfg.get("hidden_state_proj_dim", 0) or 0)
        self.device_map = str(model_cfg.get("device_map", "auto"))
        self.attn_implementation = str(model_cfg.get("attn_implementation", "")).strip() or None
        self.record_hidden_logits_probe = bool(model_cfg.get("record_hidden_logits_probe", False))

        self.torch_compile_enabled = bool(model_cfg.get("torch_compile", False))
        self.torch_compile_mode = str(model_cfg.get("torch_compile_mode", "")).strip() or None
        self.torch_compile_fullgraph = (
            bool(model_cfg["torch_compile_fullgraph"]) if "torch_compile_fullgraph" in model_cfg else None
        )
        self.torch_compile_dynamic = (
            bool(model_cfg["torch_compile_dynamic"]) if "torch_compile_dynamic" in model_cfg else None
        )

        dtype_name = str(model_cfg.get("dtype", "float16")).lower()
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_name, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )
        chat_mode = str(model_cfg.get("chat_template_mode", "auto")).strip().lower()
        self.chat_template_mode = chat_mode if chat_mode in {"auto", "on", "off"} else "auto"
        self.chat_template_add_generation_prompt = bool(model_cfg.get("chat_template_add_generation_prompt", True))
        self.chat_system_prompt = str(model_cfg.get("chat_system_prompt", "")).strip()
        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and bool(
            getattr(self.tokenizer, "chat_template", None)
        )
        if self.chat_template_mode == "on":
            self.chat_template_enabled = has_chat_template
        elif self.chat_template_mode == "off":
            self.chat_template_enabled = False
        else:
            self.chat_template_enabled = has_chat_template
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **load_kwargs)
        self.model.eval()
        if "dynamic_use_kv_cache" in model_cfg:
            self.dynamic_use_kv_cache = bool(model_cfg.get("dynamic_use_kv_cache"))
        else:
            self.dynamic_use_kv_cache = True
        self.torch_compile_status = "disabled"
        if self.torch_compile_enabled and self.device_map.lower() != "auto":
            if bool(model_cfg.get("torch_compile_disable_cudagraphs", True)):
                os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
            compile_kwargs: Dict[str, Any] = {}
            if self.torch_compile_mode:
                compile_kwargs["mode"] = self.torch_compile_mode
            if self.torch_compile_fullgraph is not None:
                compile_kwargs["fullgraph"] = self.torch_compile_fullgraph
            if self.torch_compile_dynamic is not None:
                compile_kwargs["dynamic"] = self.torch_compile_dynamic
            self.model.forward = torch.compile(self.model.forward, **compile_kwargs)  # type: ignore[assignment]
            self.torch_compile_status = "enabled"
        elif self.torch_compile_enabled:
            self.torch_compile_enabled = False
            self.torch_compile_status = "skipped_device_map_auto"

    def _render_prompts(self, prompts: List[str]) -> List[str]:
        if not self.chat_template_enabled:
            return prompts
        rendered: List[str] = []
        for prompt in prompts:
            messages: List[Dict[str, str]] = []
            if self.chat_system_prompt:
                messages.append({"role": "system", "content": self.chat_system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=self.chat_template_add_generation_prompt,
                )
                rendered.append(str(chat_text))
            except Exception:
                rendered.append(prompt)
        return rendered

    def _probe_hidden_logits(
        self,
        enc: Dict[str, Any],
        input_lens: List[int],
    ) -> List[Dict[str, Any]]:
        with self._torch.no_grad():
            fw = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        logits = fw.logits
        last_hidden = fw.hidden_states[-1] if fw.hidden_states is not None else None
        probes: List[Dict[str, Any]] = []
        for i, input_len in enumerate(input_lens):
            last_idx = max(0, int(input_len) - 1)
            lv = logits[i, last_idx].float()
            top_k = 2 if int(lv.shape[-1]) >= 2 else 1
            top_vals, top_ids = self._torch.topk(lv, k=top_k)
            probs = self._torch.softmax(lv, dim=-1)
            entropy = float(-(probs * self._torch.log(probs.clamp_min(1e-12))).sum().item())
            margin = float((top_vals[0] - top_vals[1]).item()) if top_k >= 2 else 0.0
            prob_top_vals, _ = self._torch.topk(probs, k=top_k)
            top1_prob = float(prob_top_vals[0].item()) if top_k >= 1 else 0.0
            top2_prob = float(prob_top_vals[1].item()) if top_k >= 2 else 0.0
            mass_k = min(8, int(probs.shape[-1]))
            topk_mass = float(self._torch.topk(probs, k=max(1, mass_k)).values.sum().item())
            eos_prob = 0.0
            eos_rank = float(int(probs.shape[-1]) + 1)
            if self.tokenizer.eos_token_id is not None:
                eos_id = int(self.tokenizer.eos_token_id)
                if 0 <= eos_id < int(probs.shape[-1]):
                    eos_prob = float(probs[eos_id].item())
                    eos_rank = float(1 + int((probs > probs[eos_id]).sum().item()))
            hidden_norm = (
                float(last_hidden[i, last_idx].float().norm().item()) if last_hidden is not None else None
            )
            hidden_proj = (
                list(
                    compress_hidden_state(
                        last_hidden[i, last_idx].float().detach().cpu().numpy(),
                        int(self.hidden_state_proj_dim),
                    )
                )
                if last_hidden is not None and int(self.hidden_state_proj_dim) > 0
                else []
            )
            probes.append(
                {
                    "entropy": entropy,
                    "margin": margin,
                    "top1_token_id": int(top_ids[0].item()),
                    "top1_logit": float(top_vals[0].item()),
                    "top1_prob": top1_prob,
                    "top2_prob": top2_prob,
                    "topk_mass": topk_mass,
                    "eos_prob": eos_prob,
                    "eos_rank": eos_rank,
                    "repeat_ngram_ratio": 0.0,
                    "hidden_l2_norm": hidden_norm,
                    "hidden_state_proj": hidden_proj,
                }
            )
        return probes

    def _build_gen_kwargs(self, *, info_mode: int, cot_mode: int) -> Dict[str, Any]:
        info = INFO_MODE_TABLE[info_mode]
        cot = COT_MODE_TABLE[cot_mode]
        max_new_tokens = cot_thinking_token_budget(
            cot,
            max_new_tokens_per_step=self.max_new_tokens_per_step,
        )
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            # AWQ checkpoints can occasionally produce invalid logits under sampling;
            # keep generation robust by filtering invalid values before token sampling.
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }
        if info.temperature <= 0:
            gen_kwargs["do_sample"] = False
            # Normalize sampling knobs in greedy mode to avoid repeated generation warnings
            # and reduce per-call logging overhead.
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 50
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(info.temperature)
            gen_kwargs["top_p"] = float(info.top_p)
        return gen_kwargs

    def _entropy_margin_from_logits(self, logits_vec: Any) -> tuple[float, float]:
        lv = logits_vec.float()
        top_k = 2 if int(lv.shape[-1]) >= 2 else 1
        top_vals, _ = self._torch.topk(lv, k=top_k)
        probs = self._torch.softmax(lv, dim=-1)
        entropy = float(-(probs * self._torch.log(probs.clamp_min(1e-12))).sum().item())
        margin = float((top_vals[0] - top_vals[1]).item()) if top_k >= 2 else 0.0
        return entropy, margin

    def _token_observables(self, logits_vec: Any, generated_ids: Sequence[int]) -> Dict[str, float]:
        lv = logits_vec.float()
        probs = self._torch.softmax(lv, dim=-1)
        top_k = 2 if int(probs.shape[-1]) >= 2 else 1
        top_vals, _ = self._torch.topk(probs, k=top_k)
        top1_prob = float(top_vals[0].item()) if top_k >= 1 else 0.0
        top2_prob = float(top_vals[1].item()) if top_k >= 2 else 0.0
        mass_k = min(8, int(probs.shape[-1]))
        topk_mass = float(self._torch.topk(probs, k=max(1, mass_k)).values.sum().item())
        eos_prob = 0.0
        eos_rank = float(int(probs.shape[-1]) + 1)
        if self.tokenizer.eos_token_id is not None:
            eos_id = int(self.tokenizer.eos_token_id)
            if 0 <= eos_id < int(probs.shape[-1]):
                eos_prob = float(probs[eos_id].item())
                eos_rank = float(1 + int((probs > probs[eos_id]).sum().item()))
        return {
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "topk_mass": topk_mass,
            "eos_prob": eos_prob,
            "eos_rank": eos_rank,
            "repeat_ngram_ratio": _repeat_ngram_ratio(generated_ids, n=3, window=128),
        }

    def _sample_token_id(self, logits_vec: Any, *, info_mode: int, forbid_eos: bool = False) -> int:
        info = INFO_MODE_TABLE[int(info_mode)]
        lv = logits_vec.float()
        if bool(forbid_eos) and self.tokenizer.eos_token_id is not None:
            eos_id = int(self.tokenizer.eos_token_id)
            if 0 <= eos_id < int(lv.shape[-1]):
                lv = lv.clone()
                lv[eos_id] = float("-inf")
        if info.temperature <= 0:
            return int(self._torch.argmax(lv).item())
        lv = lv / max(1e-6, float(info.temperature))
        probs = self._torch.softmax(lv, dim=-1)
        top_p = float(max(1e-6, min(1.0, info.top_p)))
        if top_p < 0.9999:
            sorted_probs, sorted_idx = self._torch.sort(probs, descending=True)
            csum = self._torch.cumsum(sorted_probs, dim=-1)
            remove_mask = csum > top_p
            if int(remove_mask.shape[-1]) > 0:
                remove_mask[0] = False
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            denom = float(sorted_probs.sum().item())
            if denom <= 1e-12:
                return int(self._torch.argmax(lv).item())
            sorted_probs = sorted_probs / sorted_probs.sum()
            sampled_local = int(self._torch.multinomial(sorted_probs, num_samples=1).item())
            return int(sorted_idx[sampled_local].item())
        sampled = int(self._torch.multinomial(probs, num_samples=1).item())
        return sampled

    @staticmethod
    def _answer_completion_ready(*, generated_text: str, answer_trigger_text: str) -> bool:
        text = str(generated_text or "")
        if not text.strip():
            return False
        if "```" in text and text.count("```") >= 2:
            return True
        if str(answer_trigger_text).strip():
            marker = "FINAL_ANSWER:"
            upper_text = text.upper()
            marker_pos = upper_text.rfind(marker)
            if marker_pos >= 0:
                tail = text[marker_pos + len(marker) :]
                stripped_tail = tail.strip()
                return bool(stripped_tail) and (("\n" in tail) or (len(stripped_tail) >= 24))
        stripped = text.rstrip()
        return stripped.endswith((".", "?", "!", "```"))

    def _advance_context_with_tokens(
        self,
        token_ids: Sequence[int],
        *,
        past: Any,
        running_input_ids: Any,
        running_attention_mask: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        logits_vec = None
        last_hidden = None
        if not token_ids:
            return past, running_input_ids, running_attention_mask, logits_vec, last_hidden
        for token_id in token_ids:
            step_input = self._torch.tensor([[int(token_id)]], device=self.model.device)
            if self.dynamic_use_kv_cache:
                fw = self.model(
                    input_ids=step_input,
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = fw.past_key_values
                logits_vec = fw.logits[0, -1]
                last_hidden = fw.hidden_states[-1][0, -1] if fw.hidden_states is not None else None
            else:
                running_input_ids = self._torch.cat([running_input_ids, step_input], dim=1)
                running_attention_mask = self._torch.cat(
                    [
                        running_attention_mask,
                        self._torch.ones((1, 1), device=self.model.device, dtype=running_attention_mask.dtype),
                    ],
                    dim=1,
                )
                fw = self.model(
                    input_ids=running_input_ids,
                    attention_mask=running_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                logits_vec = fw.logits[0, -1]
                last_hidden = fw.hidden_states[-1][0, -1] if fw.hidden_states is not None else None
        return past, running_input_ids, running_attention_mask, logits_vec, last_hidden

    def _continue_answer_completion(
        self,
        *,
        prompt: str,
        partial_prediction: str,
        info_mode: int,
        answer_trigger_text: str,
        answer_completion_budget: int,
    ) -> tuple[str, int, Dict[str, Any]]:
        budget = max(0, int(answer_completion_budget))
        if budget <= 0:
            return "", 0, {"used": False, "reason": "disabled"}
        continuation_prompt = f"{prompt}{partial_prediction}{answer_trigger_text}"
        model_prompts = self._render_prompts([continuation_prompt])
        tok_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.input_max_length > 0:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = int(self.input_max_length)
        enc = self.tokenizer(model_prompts, **tok_kwargs)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_seq_len = int(enc["input_ids"].shape[1])
        gen_kwargs = self._build_gen_kwargs(info_mode=info_mode, cot_mode=1)
        gen_kwargs["max_new_tokens"] = int(budget)
        with self._torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        new_ids = out[0][prompt_seq_len:]
        prediction = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        token_count = int(new_ids.shape[0])
        return prediction, token_count, {
            "used": True,
            "trigger_text": str(answer_trigger_text),
            "answer_completion_budget": int(budget),
        }

    @staticmethod
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

    def generate(
        self,
        prompt: str,
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        return self.generate_batch(
            [prompt],
            info_mode=info_mode,
            cot_mode=cot_mode,
            answer_trigger_texts=[str(answer_trigger_text)],
            answer_completion_budget=answer_completion_budget,
        )[0]

    def generate_dynamic(
        self,
        prompt: str,
        *,
        initial_info_mode: int,
        initial_cot_mode: int,
        on_boundary: Callable[[Dict[str, Any]], tuple[int, int, Dict[str, Any]]],
        max_new_tokens: int | None = None,
        min_new_tokens_before_eos: int = 0,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        model_prompts = self._render_prompts([prompt])
        tok_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.input_max_length > 0:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = int(self.input_max_length)
        enc = self.tokenizer(model_prompts, **tok_kwargs)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_seq_len = int(enc["input_ids"].shape[1])
        if "attention_mask" in enc:
            input_len = int(enc["attention_mask"].sum(dim=1).tolist()[0])
        else:
            input_len = prompt_seq_len

        cot0 = COT_MODE_TABLE[int(initial_cot_mode)]
        hard_cap = int(
            max_new_tokens
            or cot_thinking_token_budget(
                cot0,
                max_new_tokens_per_step=self.max_new_tokens_per_step,
            )
        )
        hard_cap = max(8, hard_cap)
        min_no_eos = max(0, int(self.dynamic_min_new_tokens_no_eos), int(min_new_tokens_before_eos))
        answer_budget_total = max(0, int(answer_completion_budget))

        info_mode = int(initial_info_mode)
        cot_mode = int(initial_cot_mode)
        segment_total = max(
            4,
            cot_thinking_token_budget(
                COT_MODE_TABLE[cot_mode],
                max_new_tokens_per_step=self.max_new_tokens_per_step,
            )
            // 4,
        )
        segment_budget = int(segment_total)

        if self.torch_compile_enabled:
            try:
                self._torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass
        generated_ids: List[int] = []
        mode_trace: List[Dict[str, Any]] = []
        decoded_tail = ""
        generated_text = ""
        tokens_since_boundary = 0
        segment_obs: Dict[str, List[float]] = {k: [] for k in OBSERVABLE_KEYS}
        thinking_tokens_remaining = int(hard_cap)
        answer_budget_remaining = int(answer_budget_total)
        in_answer_phase = False
        forced_answer_trigger_applied = False
        forced_trigger_insert_at: int | None = None

        with self._torch.no_grad():
            running_attention_mask = enc.get("attention_mask")
            if running_attention_mask is None:
                running_attention_mask = self._torch.ones(
                    (1, int(enc["input_ids"].shape[1])),
                    device=self.model.device,
                    dtype=self._torch.long,
                )
            if self.dynamic_use_kv_cache:
                fw = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=running_attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
                past = fw.past_key_values
                running_input_ids = None
            else:
                running_input_ids = enc["input_ids"]
                fw = self.model(
                    input_ids=running_input_ids,
                    attention_mask=running_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                past = None
            logits_vec = fw.logits[0, input_len - 1]
            last_hidden = fw.hidden_states[-1][0, input_len - 1] if fw.hidden_states is not None else None

            while True:
                if not in_answer_phase and thinking_tokens_remaining <= 0:
                    if answer_budget_remaining <= 0:
                        break
                    if str(answer_trigger_text).strip():
                        forced_ids = self.tokenizer.encode(str(answer_trigger_text), add_special_tokens=False)
                        past, running_input_ids, running_attention_mask, logits_vec, last_hidden = self._advance_context_with_tokens(
                            forced_ids,
                            past=past,
                            running_input_ids=running_input_ids,
                            running_attention_mask=running_attention_mask,
                        )
                        generated_text += str(answer_trigger_text)
                        decoded_tail = (decoded_tail + str(answer_trigger_text))[-128:]
                        forced_answer_trigger_applied = True
                        if forced_trigger_insert_at is None:
                            forced_trigger_insert_at = len(generated_ids)
                    in_answer_phase = True
                    continue
                if in_answer_phase and answer_budget_remaining <= 0:
                    break
                token_id = self._sample_token_id(
                    logits_vec,
                    info_mode=info_mode,
                    forbid_eos=((not in_answer_phase) and (len(generated_ids) < min_no_eos)),
                )
                generated_ids.append(int(token_id))
                piece = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
                generated_text += str(piece)
                decoded_tail = (decoded_tail + str(piece))[-128:]
                if in_answer_phase:
                    answer_budget_remaining -= 1
                else:
                    thinking_tokens_remaining -= 1
                    segment_budget -= 1
                    tokens_since_boundary += 1
                obs = self._token_observables(logits_vec, generated_ids)
                for key in OBSERVABLE_KEYS:
                    segment_obs[key].append(float(obs.get(key, 0.0)))

                eos_reached = (
                    self.tokenizer.eos_token_id is not None and int(token_id) == int(self.tokenizer.eos_token_id)
                )
                entropy, margin = self._entropy_margin_from_logits(logits_vec)
                boundary_kind = self._boundary_kind(decoded_tail)
                boundary = False
                if not in_answer_phase:
                    boundary = eos_reached or segment_budget <= 0 or (
                        tokens_since_boundary >= max(1, int(self.dynamic_boundary_min_tokens)) and bool(boundary_kind)
                    )
                elif eos_reached:
                    boundary = True
                if boundary:
                    marker = "FINAL_ANSWER:"
                    upper_text = generated_text.upper()
                    marker_pos = upper_text.rfind(marker)
                    final_answer_ready = False
                    if marker_pos >= 0:
                        tail = generated_text[marker_pos + len(marker) :]
                        final_answer_ready = ("\n" in tail) or (len(tail.strip()) >= 24)
                    code_block_closed = generated_text.count("```") >= 2
                    if eos_reached:
                        boundary_kind = "eos"
                    elif final_answer_ready:
                        boundary_kind = "answer_ready"
                    elif code_block_closed:
                        boundary_kind = "code_ready"
                    elif segment_budget <= 0 and not boundary_kind:
                        boundary_kind = "segment_budget"
                    segment_progress_ratio = min(
                        1.0,
                        float(tokens_since_boundary) / float(max(1, int(segment_total))),
                    )
                    agg = {k: _aggregate_series(vs) for k, vs in segment_obs.items()}
                    state = {
                        "entropy": float(entropy),
                        "margin": float(margin),
                        "top1_prob": float(agg["top1_prob"]["last"]),
                        "top2_prob": float(agg["top2_prob"]["last"]),
                        "topk_mass": float(agg["topk_mass"]["last"]),
                        "eos_prob": float(agg["eos_prob"]["last"]),
                        "eos_rank": float(agg["eos_rank"]["last"]),
                        "repeat_ngram_ratio": float(agg["repeat_ngram_ratio"]["last"]),
                        "boundary_observables": agg,
                        "prompt_len": len(prompt) + len(decoded_tail),
                        "generated_tokens": len(generated_ids),
                        "current_info_mode": int(info_mode),
                        "current_cot_mode": int(cot_mode),
                        "decoded_tail": decoded_tail,
                        "is_boundary": True,
                        "eos_reached": bool(eos_reached),
                        "contains_final_answer_marker": bool(marker_pos >= 0),
                        "final_answer_ready": bool(final_answer_ready),
                        "code_block_closed": bool(code_block_closed),
                        "hidden_l2_norm": float(last_hidden.float().norm().item()) if last_hidden is not None else 0.0,
                        "hidden_state_proj": list(
                            compress_hidden_state(
                                last_hidden.float().detach().cpu().numpy(),
                                int(self.hidden_state_proj_dim),
                            )
                        )
                        if last_hidden is not None and int(self.hidden_state_proj_dim) > 0
                        else [],
                        "boundary_kind": boundary_kind or "segment_budget",
                        "segment_progress_ratio": segment_progress_ratio,
                        "segment_budget_total": int(segment_total),
                        "segment_budget_remaining": max(0, int(segment_budget)),
                    }
                    if in_answer_phase:
                        if eos_reached or self._answer_completion_ready(
                            generated_text=generated_text,
                            answer_trigger_text=answer_trigger_text,
                        ):
                            break
                    else:
                        new_info, new_cot, extra = on_boundary(state)
                        info_mode, cot_mode = int(new_info), int(new_cot)
                        tokens_since_boundary = 0
                        segment_obs = {k: [] for k in OBSERVABLE_KEYS}
                        segment_total = max(
                            4,
                            cot_thinking_token_budget(
                                COT_MODE_TABLE[cot_mode],
                                max_new_tokens_per_step=self.max_new_tokens_per_step,
                            )
                            // 4,
                        )
                        segment_budget = int(segment_total)
                        mode_trace.append(
                            {
                                "token_index": len(generated_ids),
                                "info_mode": int(info_mode),
                                "cot_mode": int(cot_mode),
                                "entropy": float(entropy),
                                "margin": float(margin),
                                "top1_prob": float(state["top1_prob"]),
                                "top2_prob": float(state["top2_prob"]),
                                "topk_mass": float(state["topk_mass"]),
                                "eos_prob": float(state["eos_prob"]),
                                "eos_rank": float(state["eos_rank"]),
                                "repeat_ngram_ratio": float(state["repeat_ngram_ratio"]),
                                "boundary_observables": dict(state["boundary_observables"]),
                                "hidden_l2_norm": state["hidden_l2_norm"],
                                "hidden_state_proj": list(state["hidden_state_proj"]),
                                "boundary_kind": state["boundary_kind"],
                                "segment_progress_ratio": state["segment_progress_ratio"],
                                "extra": extra,
                            }
                        )
                        if eos_reached:
                            break
                        if cot_mode == 0:
                            if answer_budget_remaining <= 0:
                                break
                            if str(answer_trigger_text).strip():
                                forced_ids = self.tokenizer.encode(str(answer_trigger_text), add_special_tokens=False)
                                past, running_input_ids, running_attention_mask, logits_vec, last_hidden = self._advance_context_with_tokens(
                                    forced_ids,
                                    past=past,
                                    running_input_ids=running_input_ids,
                                    running_attention_mask=running_attention_mask,
                                )
                                generated_text += str(answer_trigger_text)
                                decoded_tail = (decoded_tail + str(answer_trigger_text))[-128:]
                                forced_answer_trigger_applied = True
                                if forced_trigger_insert_at is None:
                                    forced_trigger_insert_at = len(generated_ids)
                            in_answer_phase = True
                            continue

                if in_answer_phase and self._answer_completion_ready(
                    generated_text=generated_text,
                    answer_trigger_text=answer_trigger_text,
                ):
                    break

                if eos_reached:
                        break

                step_input = self._torch.tensor([[int(token_id)]], device=self.model.device)
                if self.dynamic_use_kv_cache:
                    fw = self.model(
                        input_ids=step_input,
                        use_cache=True,
                        past_key_values=past,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    past = fw.past_key_values
                    logits_vec = fw.logits[0, -1]
                    last_hidden = fw.hidden_states[-1][0, -1] if fw.hidden_states is not None else None
                else:
                    running_input_ids = self._torch.cat([running_input_ids, step_input], dim=1)
                    running_attention_mask = self._torch.cat(
                        [
                            running_attention_mask,
                            self._torch.ones((1, 1), device=self.model.device, dtype=running_attention_mask.dtype),
                        ],
                        dim=1,
                    )
                    fw = self.model(
                        input_ids=running_input_ids,
                        attention_mask=running_attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits_vec = fw.logits[0, -1]
                    last_hidden = fw.hidden_states[-1][0, -1] if fw.hidden_states is not None else None

        prediction = _decode_prediction_with_forced_trigger(
            self.tokenizer,
            generated_ids=generated_ids,
            answer_trigger_text=answer_trigger_text,
            forced_trigger_insert_at=forced_trigger_insert_at,
        )
        return GenerationResult(
            prediction=prediction,
            token_count=int(len(generated_ids)),
            raw={
                "backend": "transformers",
                "dynamic": True,
                "batch_size": 1,
                "prompt_seq_len": prompt_seq_len,
                "mode_trace": mode_trace,
                "final_info_mode": int(info_mode),
                "final_cot_mode": int(cot_mode),
                "attn_implementation": self.attn_implementation,
                "chat_template_enabled": self.chat_template_enabled,
                "chat_template_mode": self.chat_template_mode,
                "torch_compile": self.torch_compile_enabled,
                "torch_compile_status": self.torch_compile_status,
                "dynamic_use_kv_cache": self.dynamic_use_kv_cache,
                "dynamic_min_new_tokens_no_eos": int(min_no_eos),
                "answer_completion_budget": int(answer_budget_total),
                "forced_answer_trigger_applied": bool(forced_answer_trigger_applied),
                "in_answer_phase": bool(in_answer_phase),
            },
        )

    def probe_prompts(self, prompts: List[str]) -> List[Dict[str, Any]]:
        if not prompts:
            return []
        model_prompts = self._render_prompts(prompts)
        tok_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.input_max_length > 0:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = int(self.input_max_length)
        enc = self.tokenizer(model_prompts, **tok_kwargs)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        if "attention_mask" in enc:
            input_lens = enc["attention_mask"].sum(dim=1).tolist()
        else:
            input_lens = [int(enc["input_ids"].shape[1])] * len(prompts)
        return self._probe_hidden_logits(enc=enc, input_lens=input_lens)

    def generate_batch(
        self,
        prompts: List[str],
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_texts: Sequence[str] | None = None,
        answer_completion_budget: int = 0,
    ) -> List[GenerationResult]:
        if not prompts:
            return []
        if answer_trigger_texts is None:
            answer_trigger_texts = [""] * len(prompts)
        elif len(answer_trigger_texts) != len(prompts):
            raise ValueError("answer_trigger_texts length must match prompts length.")
        gen_kwargs = self._build_gen_kwargs(info_mode=info_mode, cot_mode=cot_mode)
        model_prompts = self._render_prompts(prompts)

        tok_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.input_max_length > 0:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = int(self.input_max_length)
        enc = self.tokenizer(model_prompts, **tok_kwargs)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_seq_len = int(enc["input_ids"].shape[1])
        if "attention_mask" in enc:
            input_lens = enc["attention_mask"].sum(dim=1).tolist()
        else:
            input_lens = [prompt_seq_len] * len(prompts)
        probe_stats: List[Dict[str, Any]] | None = None
        if self.record_hidden_logits_probe:
            probe_stats = self._probe_hidden_logits(enc=enc, input_lens=input_lens)

        if self.torch_compile_enabled:
            try:
                self._torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass
        with self._torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        pad_id = self.tokenizer.pad_token_id
        info = INFO_MODE_TABLE[info_mode]
        cot = COT_MODE_TABLE[cot_mode]
        max_new_tokens = cot_thinking_token_budget(
            cot,
            max_new_tokens_per_step=self.max_new_tokens_per_step,
        )

        results: List[GenerationResult] = []
        for i, input_len in enumerate(input_lens):
            # With left padding, slicing by non-pad length (`input_len`) leaks prompt tail
            # into "new tokens". The generated continuation always starts after the padded
            # input sequence width (`prompt_seq_len`).
            new_ids = out[i][prompt_seq_len:]
            if pad_id is not None and int(new_ids.numel()) > 0:
                non_pad = (new_ids != pad_id).nonzero(as_tuple=True)[0]
                if int(non_pad.numel()) == 0:
                    new_ids = new_ids[:0]
                else:
                    new_ids = new_ids[: int(non_pad[-1]) + 1]
            prediction = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            token_count = int(new_ids.shape[0])
            answer_completion_used = False
            answer_meta: Dict[str, Any] = {"used": False}
            if (
                int(answer_completion_budget) > 0
                and token_count >= int(max_new_tokens)
                and not self._answer_completion_ready(
                    generated_text=prediction,
                    answer_trigger_text=str(answer_trigger_texts[i]),
                )
            ):
                continuation, extra_tokens, answer_meta = self._continue_answer_completion(
                    prompt=str(prompts[i]),
                    partial_prediction=prediction,
                    info_mode=info_mode,
                    answer_trigger_text=str(answer_trigger_texts[i]),
                    answer_completion_budget=int(answer_completion_budget),
                )
                if continuation:
                    prediction = f"{prediction}{continuation}"
                token_count += int(extra_tokens)
                answer_completion_used = bool(answer_meta.get("used", False))
            results.append(
                GenerationResult(
                    prediction=prediction,
                    token_count=token_count,
                    raw={
                        "backend": "transformers",
                        "batch_size": len(prompts),
                        "max_new_tokens": max_new_tokens,
                        "temperature": info.temperature,
                        "top_p": info.top_p,
                        "attn_implementation": self.attn_implementation,
                        "chat_template_enabled": self.chat_template_enabled,
                        "chat_template_mode": self.chat_template_mode,
                        "torch_compile": self.torch_compile_enabled,
                        "torch_compile_status": self.torch_compile_status,
                        "hidden_logits_probe": (
                            probe_stats[i] if probe_stats is not None and i < len(probe_stats) else None
                        ),
                        "answer_completion_budget": int(answer_completion_budget),
                        "answer_completion_used": bool(answer_completion_used),
                        "answer_completion": answer_meta,
                    },
                )
            )
        return results


class VLLMCustomBackend:
    """vLLM-based backend for high-throughput offline batch inference (data collection)."""

    # Number of top logprobs to request from vLLM per token.
    # Default vLLM max_logprobs=20, but we raise it to 50 via engine args
    # for better entropy approximation and full top-k coverage.
    LOGPROBS_K: int = 50

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "vLLM backend requires `vllm`. Install with `pip install vllm`."
            ) from exc

        from transformers import AutoTokenizer
        import vllm
        from vllm import LLM, SamplingParams
        try:
            from vllm.config import ReasoningConfig  # type: ignore
        except Exception:
            ReasoningConfig = None  # type: ignore

        self._SamplingParams = SamplingParams
        self.vllm_version = str(getattr(vllm, "__version__", "unknown"))
        self.model_dir = str(model_cfg["model_dir"])
        self.tokenizer_dir = str(model_cfg.get("tokenizer_dir", self.model_dir))
        self.download_dir = str(model_cfg.get("download_dir", "")).strip() or None
        self.cache_dir = str(model_cfg.get("cache_dir", self.download_dir or "")).strip() or None
        hf_home = str(model_cfg.get("hf_home", "")).strip()
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
            os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))
        self.trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        self.max_new_tokens_per_step = int(model_cfg.get("max_new_tokens_per_step", 16))
        self.collect_logprobs = bool(model_cfg.get("collect_logprobs", True))
        self.sampling_seed = model_cfg.get("sampling_seed")
        self._sampling_params_supports_seed = "seed" in inspect.signature(SamplingParams).parameters
        self._sampling_params_supports_thinking_token_budget = (
            "thinking_token_budget" in inspect.signature(SamplingParams).parameters
        )
        raw_reasoning_config = model_cfg.get("reasoning_config")
        self.reasoning_config = None
        if raw_reasoning_config:
            if ReasoningConfig is not None and isinstance(raw_reasoning_config, dict):
                rc_dict = dict(raw_reasoning_config)
                # Support both documented key shapes seen across vLLM docs/pages.
                if "think_start_str" in rc_dict and "reasoning_start_str" not in rc_dict:
                    rc_dict["reasoning_start_str"] = rc_dict.pop("think_start_str")
                if "think_end_str" in rc_dict and "reasoning_end_str" not in rc_dict:
                    rc_dict["reasoning_end_str"] = rc_dict.pop("think_end_str")
                self.reasoning_config = ReasoningConfig(**rc_dict)
            else:
                self.reasoning_config = raw_reasoning_config
        self._native_thinking_budget_enabled = bool(
            self._sampling_params_supports_thinking_token_budget and self.reasoning_config
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )
        chat_mode = str(model_cfg.get("chat_template_mode", "auto")).strip().lower()
        self.chat_template_mode = chat_mode if chat_mode in {"auto", "on", "off"} else "auto"
        self.chat_template_add_generation_prompt = bool(model_cfg.get("chat_template_add_generation_prompt", True))
        self.chat_system_prompt = str(model_cfg.get("chat_system_prompt", "")).strip()
        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and bool(
            getattr(self.tokenizer, "chat_template", None)
        )
        if self.chat_template_mode == "on":
            self.chat_template_enabled = has_chat_template
        elif self.chat_template_mode == "off":
            self.chat_template_enabled = False
        else:
            self.chat_template_enabled = has_chat_template

        llm_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": str(model_cfg.get("dtype", "float16")).lower(),
        }
        if model_cfg.get("max_model_len"):
            llm_kwargs["max_model_len"] = int(model_cfg["max_model_len"])
        if model_cfg.get("quantization"):
            llm_kwargs["quantization"] = str(model_cfg["quantization"])
        if model_cfg.get("tensor_parallel_size"):
            llm_kwargs["tensor_parallel_size"] = int(model_cfg["tensor_parallel_size"])
        if model_cfg.get("gpu_memory_utilization") is not None:
            llm_kwargs["gpu_memory_utilization"] = float(model_cfg["gpu_memory_utilization"])
        if self.download_dir:
            llm_kwargs["download_dir"] = self.download_dir
        if self.collect_logprobs:
            llm_kwargs["max_logprobs"] = self.LOGPROBS_K
        if self.reasoning_config:
            llm_kwargs["reasoning_config"] = self.reasoning_config

        self.llm = LLM(model=self.model_dir, **llm_kwargs)

    def _render_prompts(self, prompts: List[str]) -> List[str]:
        if not self.chat_template_enabled:
            return prompts
        rendered: List[str] = []
        for prompt in prompts:
            messages: List[Dict[str, str]] = []
            if self.chat_system_prompt:
                messages.append({"role": "system", "content": self.chat_system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=self.chat_template_add_generation_prompt,
                )
                rendered.append(str(chat_text))
            except Exception:
                rendered.append(prompt)
        return rendered

    def generate(
        self,
        prompt: str,
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        return self.generate_batch(
            [prompt],
            info_mode=info_mode,
            cot_mode=cot_mode,
            answer_trigger_texts=[str(answer_trigger_text)],
            answer_completion_budget=answer_completion_budget,
        )[0]

    def generate_batch(
        self,
        prompts: List[str],
        *,
        info_mode: int,
        cot_mode: int,
        answer_trigger_texts: Sequence[str] | None = None,
        answer_completion_budget: int = 0,
    ) -> List[GenerationResult]:
        if not prompts:
            return []
        if answer_trigger_texts is not None and len(answer_trigger_texts) != len(prompts):
            raise ValueError("answer_trigger_texts length must match prompts length.")

        info = INFO_MODE_TABLE[info_mode]
        cot = COT_MODE_TABLE[cot_mode]
        thinking_budget = cot_thinking_token_budget(
            cot,
            max_new_tokens_per_step=self.max_new_tokens_per_step,
        )
        answer_budget = max(0, int(answer_completion_budget))
        total_max_tokens = thinking_budget

        if self._native_thinking_budget_enabled:
            total_max_tokens = thinking_budget + answer_budget
            sp_kwargs: Dict[str, Any] = {
                "max_tokens": max(1, total_max_tokens),
                "thinking_token_budget": int(thinking_budget),
            }
        else:
            sp_kwargs = {"max_tokens": int(thinking_budget)}
        if info.temperature <= 0:
            sp_kwargs["temperature"] = 0.0
        else:
            sp_kwargs["temperature"] = float(info.temperature)
            sp_kwargs["top_p"] = float(info.top_p)
        if self.collect_logprobs:
            sp_kwargs["logprobs"] = self.LOGPROBS_K
        if self._sampling_params_supports_seed and self.sampling_seed is not None:
            sp_kwargs["seed"] = int(self.sampling_seed)

        sampling_params = self._SamplingParams(**sp_kwargs)
        model_prompts = self._render_prompts(prompts)
        prompt_token_counts = [
            len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
            for prompt_text in model_prompts
        ]
        outputs = self.llm.generate(model_prompts, sampling_params)

        results: List[GenerationResult] = []
        for prompt_text, answer_trigger_text, output, prompt_token_count in zip(
            prompts,
            (answer_trigger_texts or [""] * len(prompts)),
            outputs,
            prompt_token_counts,
        ):
            comp = output.outputs[0]
            generated_text = comp.text
            token_count = len(comp.token_ids)
            answer_completion_used = False
            answer_meta: Dict[str, Any] = {"used": False}
            raw: Dict[str, Any] = {
                "backend": "vllm",
                "vllm_version": self.vllm_version,
                "batch_size": len(prompts),
                "max_new_tokens": thinking_budget,
                "thinking_token_budget": int(thinking_budget),
                "max_total_tokens": int(total_max_tokens),
                "temperature": info.temperature,
                "top_p": info.top_p,
                "temperature_actual": float(info.temperature),
                "top_p_actual": float(info.top_p),
                "top_k_actual": None,
                "info_mode": info_mode,
                "cot_mode": cot_mode,
                "sampling_seed": (int(self.sampling_seed) if self.sampling_seed is not None else None),
                "prompt_token_count": int(prompt_token_count),
                "answer_completion_budget": int(answer_budget),
                "native_thinking_token_budget": bool(self._native_thinking_budget_enabled),
                "finish_reason": getattr(comp, "finish_reason", None),
                "stop_reason": getattr(comp, "stop_reason", None),
                "stop_token_ids": list(getattr(comp, "stop_token_ids", []) or []),
            }
            if self.collect_logprobs and comp.logprobs:
                raw["_token_ids"] = [int(token_id) for token_id in comp.token_ids]
                raw["_logprobs_list"] = self._serialize_logprobs(comp.logprobs)
                raw["token_logprobs"] = self._extract_token_features(comp, token_count)
            if (
                not self._native_thinking_budget_enabled
                and answer_budget > 0
                and token_count >= int(thinking_budget)
                and not TransformersCustomBackend._answer_completion_ready(
                    generated_text=generated_text,
                    answer_trigger_text=str(answer_trigger_text),
                )
            ):
                continuation, extra_tokens, answer_meta = self._continue_answer_completion(
                    prompt=str(prompt_text),
                    partial_prediction=generated_text,
                    info_mode=info_mode,
                    answer_trigger_text=str(answer_trigger_text),
                    answer_completion_budget=answer_budget,
                )
                if continuation:
                    generated_text = f"{generated_text}{continuation}"
                token_count += int(extra_tokens)
                answer_completion_used = bool(answer_meta.get("used", False))
                raw["max_total_tokens"] = int(thinking_budget + max(0, int(extra_tokens)))
                if answer_completion_used and answer_meta.get("token_ids"):
                    raw.setdefault("_token_ids", [])
                    raw.setdefault("_logprobs_list", [])
                    raw["_token_ids"] = list(raw["_token_ids"]) + list(answer_meta.get("token_ids", []))
                    raw["_logprobs_list"] = list(raw["_logprobs_list"]) + list(
                        answer_meta.get("logprobs_list", [])
                    )
                    if self.collect_logprobs and raw.get("_logprobs_list"):
                        raw["token_logprobs"] = self._extract_token_features_from_serialized(
                            token_ids=raw["_token_ids"],
                            logprobs_list=raw["_logprobs_list"],
                        )
            raw["answer_completion_used"] = bool(answer_completion_used)
            raw["answer_completion"] = answer_meta
            results.append(
                GenerationResult(
                    prediction=generated_text,
                    token_count=token_count,
                    raw=raw,
                )
            )
        return results

    def _continue_answer_completion(
        self,
        *,
        prompt: str,
        partial_prediction: str,
        info_mode: int,
        answer_trigger_text: str,
        answer_completion_budget: int,
    ) -> tuple[str, int, Dict[str, Any]]:
        budget = max(0, int(answer_completion_budget))
        if budget <= 0:
            return "", 0, {"used": False, "reason": "disabled"}

        info = INFO_MODE_TABLE[info_mode]
        continuation_prompt = f"{prompt}{partial_prediction}{answer_trigger_text}"
        model_prompts = self._render_prompts([continuation_prompt])
        sp_kwargs: Dict[str, Any] = {"max_tokens": int(budget)}
        if info.temperature <= 0:
            sp_kwargs["temperature"] = 0.0
        else:
            sp_kwargs["temperature"] = float(info.temperature)
            sp_kwargs["top_p"] = float(info.top_p)
        if self.collect_logprobs:
            sp_kwargs["logprobs"] = self.LOGPROBS_K
        if self._sampling_params_supports_seed and self.sampling_seed is not None:
            sp_kwargs["seed"] = int(self.sampling_seed)

        sampling_params = self._SamplingParams(**sp_kwargs)
        output = self.llm.generate(model_prompts, sampling_params)[0]
        comp = output.outputs[0]
        continuation = str(comp.text or "")
        token_ids = [int(token_id) for token_id in getattr(comp, "token_ids", [])]
        logprobs_list = self._serialize_logprobs(comp.logprobs) if self.collect_logprobs and comp.logprobs else []
        return continuation, len(token_ids), {
            "used": True,
            "trigger_text": str(answer_trigger_text),
            "answer_completion_budget": int(budget),
            "token_ids": token_ids,
            "logprobs_list": logprobs_list,
            "finish_reason": getattr(comp, "finish_reason", None),
            "stop_reason": getattr(comp, "stop_reason", None),
            "stop_token_ids": list(getattr(comp, "stop_token_ids", []) or []),
        }

    @staticmethod
    def _serialize_logprobs(logprobs_list: Any) -> List[Dict[int, float]]:
        out: List[Dict[int, float]] = []
        if not logprobs_list:
            return out
        for step_logprobs in logprobs_list:
            if not step_logprobs:
                out.append({})
                continue
            step: Dict[int, float] = {}
            for tid, lp in step_logprobs.items():
                step[int(tid)] = float(lp.logprob)
            out.append(step)
        return out

    @staticmethod
    def _extract_token_features(
        comp: Any,
        token_count: int,
    ) -> List[Dict[str, Any]]:
        """Extract per-token logits-derived features from vLLM logprobs output.

        Each element in comp.logprobs is a dict mapping token_id -> Logprob.
        We compute the same features that TransformersCustomBackend collects
        at boundaries: entropy, margin, top1/2 prob, topk_mass, eos_prob.
        """
        import math as _math

        features: List[Dict[str, Any]] = []
        if not comp.logprobs:
            return features
        for step_idx, step_logprobs in enumerate(comp.logprobs):
            if not step_logprobs:
                features.append({})
                continue
            # step_logprobs: Dict[int, Logprob] with .logprob field
            # Convert to sorted (token_id, logprob) pairs
            items = []
            for tid, lp in step_logprobs.items():
                items.append((int(tid), float(lp.logprob)))
            # Sort by logprob descending
            items.sort(key=lambda x: x[1], reverse=True)

            # Convert logprobs to probs
            probs = [_math.exp(lp) for _, lp in items]
            total_prob = sum(probs)

            top1_prob = probs[0] if len(probs) >= 1 else 0.0
            top2_prob = probs[1] if len(probs) >= 2 else 0.0
            margin = (items[0][1] - items[1][1]) if len(items) >= 2 else 0.0

            def _mass_at(k: int) -> float:
                return float(sum(probs[: max(0, int(k))]))

            # Entropy approximation from top-K probs
            entropy = 0.0
            collision_prob = 0.0
            for p in probs:
                if p > 1e-12:
                    entropy -= p * _math.log(p)
                    collision_prob += p * p

            # EOS prob (if in the top-K)
            eos_prob = 0.0
            eos_rank = float(len(probs) + 1)
            # We don't know eos_token_id here, but it will be in the dict if present
            # The caller can look it up by token_id

            chosen_token_rank = None
            chosen_token_logprob = None
            if step_idx < len(comp.token_ids):
                chosen_tid = int(comp.token_ids[step_idx])
                for rank, (tid, lp) in enumerate(items):
                    if tid == chosen_tid:
                        chosen_token_rank = rank + 1
                        chosen_token_logprob = lp
                        break

            features.append({
                "top1_prob": top1_prob,
                "top1_token_id": items[0][0] if items else -1,
                "top2_prob": top2_prob,
                "margin": margin,
                "topk_mass": _mass_at(5),
                "topk_mass_1": _mass_at(1),
                "topk_mass_2": _mass_at(2),
                "topk_mass_5": _mass_at(5),
                "topk_mass_10": _mass_at(10),
                "topk_mass_20": _mass_at(20),
                "tail_mass_beyond_top20": max(0.0, 1.0 - _mass_at(20)),
                "entropy_approx": entropy,
                "collision_prob": collision_prob,
                "renyi_entropy_2": (-_math.log(max(collision_prob, 1e-12)) if collision_prob > 0.0 else 0.0),
                "effective_vocab_size": _math.exp(entropy),
                "logprob_gap_1_to_2": (items[0][1] - items[1][1]) if len(items) >= 2 else None,
                "logprob_gap_1_to_5": (items[0][1] - items[4][1]) if len(items) >= 5 else None,
                "logprob_gap_1_to_10": (items[0][1] - items[9][1]) if len(items) >= 10 else None,
                "chosen_token_rank": chosen_token_rank,
                "chosen_token_logprob": chosen_token_logprob,
                "n_logprobs": len(items),
                "total_prob_mass": total_prob,
            })
        return features

    @classmethod
    def _extract_token_features_from_serialized(
        cls,
        *,
        token_ids: Sequence[int],
        logprobs_list: Sequence[Dict[int, float]],
    ) -> List[Dict[str, Any]]:
        class _LP:
            def __init__(self, logprob: float) -> None:
                self.logprob = float(logprob)

        class _Comp:
            def __init__(self, token_ids: Sequence[int], logprobs_list: Sequence[Dict[int, float]]) -> None:
                self.token_ids = list(token_ids)
                self.logprobs = [{int(k): _LP(v) for k, v in step.items()} for step in logprobs_list]

        comp = _Comp(token_ids, logprobs_list)
        return cls._extract_token_features(comp, len(token_ids))

    def probe_prompts(self, prompts: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError("VLLMCustomBackend does not support probe_prompts.")

    def generate_dynamic(
        self,
        prompt: str,
        *,
        initial_info_mode: int,
        initial_cot_mode: int,
        on_boundary: Callable[[Dict[str, Any]], tuple[int, int, Dict[str, Any]]],
        max_new_tokens: int | None = None,
        min_new_tokens_before_eos: int = 0,
        answer_trigger_text: str = "",
        answer_completion_budget: int = 0,
    ) -> GenerationResult:
        raise NotImplementedError(
            "VLLMCustomBackend does not support generate_dynamic. "
            "Dynamic governance requires the custom Transformers backend."
        )


def build_custom_backend(model_cfg: Dict[str, Any], *, dry_run: bool) -> CustomBackend:
    if dry_run:
        return DryRunCustomBackend(model_cfg=model_cfg)
    backend = str(model_cfg.get("runtime_backend", ""))
    if backend == "vllm":
        return VLLMCustomBackend(model_cfg=model_cfg)
    if backend != "custom_transformers":
        raise ValueError(
            f"Unsupported runtime_backend={backend!r}. "
            "Expected `custom_transformers` or `vllm` for non-dry runs."
        )
    return TransformersCustomBackend(model_cfg=model_cfg)
