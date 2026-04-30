from __future__ import annotations

import json
import hashlib
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from .governance_model import GovernanceFeatureEncoder, load_governance_bundle
from .settings import COT_MODE_TABLE, INFO_MODE_TABLE

Action = Tuple[int, int]


@dataclass
class EmaStats:
    mean: float = 0.0
    var: float = 1.0
    initialized: bool = False

    def update(self, x: float, alpha: float) -> None:
        x = float(x)
        if not self.initialized:
            self.mean = x
            self.var = 1.0
            self.initialized = True
            return
        prev_mean = self.mean
        self.mean = (1.0 - alpha) * self.mean + alpha * x
        delta = x - prev_mean
        self.var = max(1e-6, (1.0 - alpha) * self.var + alpha * (delta * delta))

    def zscore(self, x: float) -> float:
        if not self.initialized:
            return 0.0
        return (float(x) - self.mean) / max(1e-6, math.sqrt(self.var))


@dataclass
class AdaptiveModeControllerConfig:
    ema_alpha: float = 0.08
    rolling_window: int = 128
    token_lambda: float = 0.25
    switch_cost: float = 0.0
    switch_margin: float = 0.03
    prompt_len_weight: float = 0.10
    uncertainty_temp: float = 1.0
    use_cot0: bool = False
    memory_bonus: float = 0.0
    anchor_cost: float = 0.0
    head_lr: float = 0.03
    head_scale: float = 1.0
    reward_clip: float = 1.5
    min_steps_between_switch: int = 1
    dynamic_hysteresis: bool = True
    hysteresis_base: float = 0.01
    hysteresis_switch_scale: float = 0.05
    recent_switch_window: int = 10
    enable_model_memory: bool = True
    memory_layout: str = "multi_bank_v2"
    global_memory_weight: float = 0.4
    model_memory_weight: float = 0.4
    episode_memory_weight: float = 0.2
    episode_memory_size: int = 64
    score_memory_weight: float = 0.0
    base_info_mode: int = 2
    base_cot_mode: int = 2
    easy_threshold: float = 0.0
    hard_threshold: float = 1.0
    easy_max_cot_mode: int = 3
    hard_min_cot_mode: int = 0
    hard_min_cot_until_progress: float = 0.0
    heuristic_prior_weight: float = 0.0
    answer_zone_max_info_mode: int = 3
    answer_zone_max_cot_mode: int = 1
    governance_model_path: str = ""
    hidden_proj_dim: int = 0
    feature_schema_version: str = "boundary_v2"

    # v11 strict-alignment knobs
    use_hidden_observables: bool = True
    online_memory_update_enabled: bool = True
    slot_prior_weight: float = 0.0
    memory_type: str = "legacy_action_ema"
    task_encoder_enabled: bool = False
    task_encoder_dim: int = 24
    target_slot_count: int = 15
    slot_merge_cosine_threshold: float = 0.90
    slot_diversity_weight: float = 0.02
    slot_diversity_margin: float = 0.35
    slot_match_temperature_init: float = 1.0
    slot_match_temperature_learnable: bool = True
    slot_schema_version: str = "v12_15slot"


class TaskEncoder:
    """Prompt-only task encoder for v12 15-slot memory routing."""

    TOKEN_PAT = re.compile(r"[a-z0-9_]+")
    SLOT_CANDIDATES: Tuple[str, ...] = (
        "math_arithmetic",
        "math_algebra",
        "math_geometry",
        "math_number_theory",
        "math_combinatorics_prob",
        "math_proof_competition",
        "math_symbolic_manipulation",
        "code_generation",
        "code_completion",
        "code_fix_debug",
        "code_io_parsing",
        "code_test_reasoning",
        "code_algorithm_design",
        "logic_symbolic",
        "logic_multichoice_reasoning",
        "qa_bool_fact",
        "planning_schedule",
        "planning_route",
    )

    SLOT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "math_arithmetic": ("calculate", "compute", "sum", "difference", "ratio", "percent", "word", "total"),
        "math_algebra": ("equation", "solve", "variable", "polynomial", "factor", "root", "expression"),
        "math_geometry": ("triangle", "circle", "angle", "area", "perimeter", "radius", "diameter", "geometry"),
        "math_number_theory": ("integer", "divisor", "prime", "mod", "modulo", "gcd", "lcm", "number"),
        "math_combinatorics_prob": ("probability", "combination", "permutation", "arrange", "count", "choose"),
        "math_proof_competition": ("prove", "proof", "olympiad", "aime", "competition", "show"),
        "math_symbolic_manipulation": ("simplify", "expand", "derivative", "integral", "symbolic", "formula"),
        "code_generation": ("write", "implement", "build", "create", "python", "function", "program"),
        "code_completion": ("complete", "fill", "todo", "missing", "continue", "skeleton"),
        "code_fix_debug": ("fix", "bug", "debug", "error", "issue", "patch", "repair"),
        "code_io_parsing": ("input", "output", "stdin", "stdout", "parse", "format"),
        "code_test_reasoning": ("test", "assert", "unit", "case", "edge", "verify"),
        "code_algorithm_design": ("algorithm", "complexity", "optimize", "dp", "graph", "greedy"),
        "logic_symbolic": ("if", "then", "implies", "therefore", "deduce", "symbolic", "logic"),
        "logic_multichoice_reasoning": ("option", "choice", "a)", "b)", "c)", "d)", "answer"),
        "qa_bool_fact": ("true", "false", "yes", "no", "fact", "statement", "question"),
        "planning_schedule": ("meeting", "calendar", "time", "schedule", "deadline", "slot", "attendee"),
        "planning_route": ("trip", "route", "travel", "city", "distance", "itinerary"),
    }

    def __init__(self, dim: int = 24, slot_schema_version: str = "v12_15slot") -> None:
        self.dim = max(12, int(dim))
        self.slot_schema_version = str(slot_schema_version or "v12_15slot")

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in self.TOKEN_PAT.findall(str(text).lower()) if t]

    @staticmethod
    def _stable_hash_index(token: str, size: int) -> int:
        if size <= 0:
            return 0
        digest = hashlib.md5(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], byteorder="little", signed=False)
        return int(value % int(size))

    def slot_signal(self, prompt_text: str) -> Dict[str, float]:
        text = str(prompt_text or "")
        lowered = text.lower()
        toks = self._tokenize(text)
        n_tok = max(1.0, float(len(toks)))
        out: Dict[str, float] = {}
        for slot_name in self.SLOT_CANDIDATES:
            kws = self.SLOT_KEYWORDS.get(slot_name, ())
            kw_hits = sum(1.0 for tk in toks if tk in kws)
            score = kw_hits / n_tok
            if slot_name.startswith("code_"):
                if ("```" in text) or ("def " in lowered) or ("class " in lowered):
                    score += 0.12
                if "input" in lowered and "output" in lowered:
                    score += 0.04
            if slot_name.startswith("math_"):
                if any(ch.isdigit() for ch in text):
                    score += 0.03
                if "$" in text or "\\(" in text or "\\[" in text:
                    score += 0.04
            if slot_name == "logic_multichoice_reasoning":
                options = re.findall(r"\b[A-E][\)\.\:]", text)
                score += min(0.20, 0.03 * float(len(options)))
            if slot_name == "qa_bool_fact":
                score += 0.08 if bool(re.search(r"\b(true|false|yes|no)\b", lowered)) else 0.0
            if slot_name == "planning_schedule":
                score += 0.08 if bool(re.search(r"\b(am|pm|monday|tuesday|wednesday|thursday|friday)\b", lowered)) else 0.0
            if slot_name == "planning_route":
                score += 0.08 if bool(re.search(r"\b(from|to|via|flight|train|hotel)\b", lowered)) else 0.0
            out[slot_name] = max(0.0, float(score))
        return out

    def predict_slot(self, prompt_text: str) -> str:
        scores = self.slot_signal(prompt_text)
        if not scores:
            return str(self.SLOT_CANDIDATES[0])
        return max(scores.items(), key=lambda kv: (float(kv[1]), kv[0]))[0]

    def encode(self, prompt_text: str) -> List[float]:
        text = str(prompt_text or "")
        lowered = text.lower()
        toks = self._tokenize(text)
        n_tok = max(1.0, float(len(toks)))
        slot_scores = self.slot_signal(text)
        base = [float(slot_scores.get(slot_name, 0.0)) for slot_name in self.SLOT_CANDIDATES]

        char_len = float(len(text))
        line_cnt = float(text.count("\n") + 1)
        digit_cnt = float(sum(ch.isdigit() for ch in text))
        punct_cnt = float(sum(ch in ".,;:!?()[]{}" for ch in text))
        code_hint = float(
            ("```" in text)
            or ("def " in lowered)
            or ("class " in lowered)
            or ("import " in lowered)
            or ("return " in lowered)
        )
        question_cnt = float(text.count("?"))
        option_cnt = float(len(re.findall(r"\b[A-E][\)\.\:]", text)))
        latex_hint = float(("$" in text) or ("\\(" in text) or ("\\[" in text))
        shape = [
            min(1.0, math.log1p(char_len) / 12.0),
            min(1.0, math.log1p(line_cnt) / 7.0),
            min(1.0, digit_cnt / max(1.0, char_len)),
            min(1.0, punct_cnt / max(1.0, char_len)),
            min(1.0, code_hint),
            min(1.0, question_cnt / 3.0),
            min(1.0, option_cnt / 6.0),
            min(1.0, latex_hint),
        ]

        vec = list(base) + list(shape)
        if len(vec) < self.dim:
            hash_dim = self.dim - len(vec)
            hashed = [0.0] * hash_dim
            for tk in toks:
                idx = self._stable_hash_index(tk, hash_dim)
                hashed[idx] += 1.0 / n_tok
            vec.extend(hashed)
        elif len(vec) > self.dim:
            vec = vec[: self.dim]

        norm = math.sqrt(sum(float(v) * float(v) for v in vec))
        if norm > 1e-12:
            vec = [float(v) / norm for v in vec]
        return vec

    def to_dict(self) -> Dict[str, object]:
        return {
            "dim": int(self.dim),
            "slot_schema_version": str(self.slot_schema_version),
            "slot_candidates": list(self.SLOT_CANDIDATES),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "TaskEncoder":
        return cls(
            dim=int(payload.get("dim", 24) or 24),
            slot_schema_version=str(payload.get("slot_schema_version", "v12_15slot")),
        )


class SlotMemory:
    """Task-conditioned strategy memory with v12 18->15 slot selection."""

    def __init__(
        self,
        *,
        encoder_dim: int,
        slots: Optional[Dict[str, Dict[str, object]]] = None,
        reserve_slots: Optional[Dict[str, Dict[str, object]]] = None,
        target_slot_count: int = 15,
        merge_cosine_threshold: float = 0.90,
        diversity_weight: float = 0.02,
        diversity_margin: float = 0.35,
        match_temperature_init: float = 1.0,
        match_temperature_learnable: bool = True,
        slot_schema_version: str = "v12_15slot",
    ) -> None:
        self.encoder_dim = int(max(12, encoder_dim))
        self.target_slot_count = max(4, int(target_slot_count))
        self.merge_cosine_threshold = float(merge_cosine_threshold)
        self.diversity_weight = max(0.0, float(diversity_weight))
        self.diversity_margin = float(diversity_margin)
        self.match_temperature = max(0.05, float(match_temperature_init))
        self.match_temperature_learnable = bool(match_temperature_learnable)
        self.slot_schema_version = str(slot_schema_version or "v12_15slot")
        self.slots: Dict[str, Dict[str, object]] = slots or self._default_slots(self.encoder_dim)
        self.reserve_slots: Dict[str, Dict[str, object]] = reserve_slots or {}
        self._last_merge_report: Dict[str, object] = {"merges": [], "initial_slot_count": len(self.slots), "final_slot_count": len(self.slots)}
        self._last_diversity_loss_stats: Dict[str, float] = {"before": 0.0, "after": 0.0}
        self._last_temperature_report: Dict[str, float] = {"before": self.match_temperature, "after": self.match_temperature}

    @property
    def last_merge_report(self) -> Dict[str, object]:
        return dict(self._last_merge_report)

    @property
    def last_diversity_loss_stats(self) -> Dict[str, float]:
        return dict(self._last_diversity_loss_stats)

    @property
    def final_slot_names(self) -> List[str]:
        return sorted(self.slots.keys())

    @staticmethod
    def _slot_prior_template(slot_name: str) -> Tuple[List[float], List[float]]:
        if slot_name.startswith("math_"):
            return [0.34, 0.27, 0.20, 0.12, 0.07], [0.04, 0.18, 0.37, 0.41]
        if slot_name.startswith("code_"):
            return [0.30, 0.28, 0.21, 0.13, 0.08], [0.07, 0.25, 0.39, 0.29]
        if slot_name.startswith("logic_"):
            return [0.22, 0.26, 0.28, 0.16, 0.08], [0.12, 0.30, 0.40, 0.18]
        if slot_name.startswith("qa_"):
            return [0.17, 0.24, 0.32, 0.18, 0.09], [0.24, 0.41, 0.25, 0.10]
        if slot_name.startswith("planning_"):
            return [0.20, 0.25, 0.29, 0.17, 0.09], [0.14, 0.33, 0.35, 0.18]
        return [0.20] * 5, [0.25] * 4

    @classmethod
    def _default_slots(cls, dim: int) -> Dict[str, Dict[str, object]]:
        def _unit(seed: int) -> List[float]:
            vals = [0.0] * dim
            vals[seed % dim] = 1.0
            vals[(seed * 5 + 3) % dim] = 0.45
            vals[(seed * 7 + 1) % dim] = 0.22
            n = math.sqrt(sum(v * v for v in vals))
            return [float(v / n) for v in vals]

        out: Dict[str, Dict[str, object]] = {}
        for idx, name in enumerate(TaskEncoder.SLOT_CANDIDATES):
            info_prior, cot_prior = cls._slot_prior_template(name)
            out[name] = {
                "slot_key": _unit(idx),
                "info_prior": list(info_prior),
                "cot_prior": list(cot_prior),
                "count": 0.0,
                "avg_reward": 0.0,
                "support_count": 0.0,
            }
        return out

    @staticmethod
    def _normalize_prob(vec: Sequence[float], size: int) -> List[float]:
        vals = [max(1e-6, float(x)) for x in list(vec)[:size]]
        if len(vals) < size:
            vals.extend([1.0] * (size - len(vals)))
        s = sum(vals)
        if s <= 1e-12:
            return [1.0 / float(size)] * size
        return [float(v / s) for v in vals]

    @staticmethod
    def _normalize_vec(vec: Sequence[float], size: int) -> List[float]:
        vals = [float(x) for x in list(vec)[:size]]
        if len(vals) < size:
            vals.extend([0.0] * (size - len(vals)))
        n = math.sqrt(sum(v * v for v in vals))
        if n <= 1e-12:
            vals = [0.0] * size
            vals[0] = 1.0
            return vals
        return [float(v / n) for v in vals]

    @staticmethod
    def _softmax(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
        arr = [float(x) / max(1e-6, float(temperature)) for x in xs]
        if not arr:
            return []
        m = max(arr)
        exps = [math.exp(x - m) for x in arr]
        z = sum(exps)
        if z <= 1e-12:
            return [1.0 / float(len(arr))] * len(arr)
        return [float(e / z) for e in exps]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a))
        nb = math.sqrt(sum(float(y) * float(y) for y in b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(dot / (na * nb))

    @staticmethod
    def _extract_prompt(row: Dict[str, object]) -> str:
        prompt = str(row.get("prompt", "") or "").strip()
        if prompt:
            return prompt
        sf = row.get("state_features")
        if isinstance(sf, dict):
            return str(sf.get("prompt", "") or "").strip()
        return ""

    @staticmethod
    def _extract_modes(row: Dict[str, object]) -> Tuple[int, int]:
        info_mode = int(row.get("chosen_info_mode", row.get("info_mode", 3)) or 3)
        cot_mode = int(row.get("chosen_cot_mode", row.get("cot_mode", 2)) or 2)
        return max(1, min(5, info_mode)), max(0, min(3, cot_mode))

    @staticmethod
    def _extract_reward(row: Dict[str, object]) -> float:
        correctness = float(row.get("is_correct", row.get("correctness", 0.0)) or 0.0)
        token_count = float(row.get("token_count", 0.0) or 0.0)
        if token_count <= 0.0:
            sf = row.get("state_features")
            if isinstance(sf, dict):
                token_count = float(sf.get("generated_tokens", 0.0) or 0.0)
        return float(correctness - (token_count / 256.0))

    @staticmethod
    def _context_bias(slot_name: str, context_tag: str) -> float:
        tag = str(context_tag or "").strip().lower()
        if tag == "code":
            if slot_name.startswith("code_"):
                return 0.12
            if slot_name.startswith("math_"):
                return -0.02
            return 0.0
        if slot_name.startswith("code_"):
            return -0.01
        if slot_name.startswith("logic_") or slot_name.startswith("qa_"):
            return 0.02
        if slot_name.startswith("planning_"):
            return 0.02
        return 0.0

    def retrieve(self, *, task_query: Sequence[float], context_tag: str) -> Dict[str, object]:
        names = sorted(self.slots.keys())
        sims: List[float] = []
        for name in names:
            slot = self.slots[name]
            key = [float(x) for x in slot.get("slot_key", [])]
            sim = self._cosine(task_query, key) + self._context_bias(name, context_tag)
            sims.append(sim)

        mix = self._softmax(sims, temperature=float(self.match_temperature))
        info = [0.0] * 5
        cot = [0.0] * 4
        for w, name in zip(mix, names):
            slot = self.slots[name]
            ip = self._normalize_prob(slot.get("info_prior", []), 5)
            cp = self._normalize_prob(slot.get("cot_prior", []), 4)
            for i in range(5):
                info[i] += float(w) * float(ip[i])
            for i in range(4):
                cot[i] += float(w) * float(cp[i])
        info = self._normalize_prob(info, 5)
        cot = self._normalize_prob(cot, 4)
        return {
            "slot_names": names,
            "slot_weights": mix,
            "info_prior": info,
            "cot_prior": cot,
            "match_temperature": float(self.match_temperature),
        }

    def action_bonus(self, *, info_prior: Sequence[float], cot_prior: Sequence[float]) -> Dict[Action, float]:
        ip = self._normalize_prob(info_prior, 5)
        cp = self._normalize_prob(cot_prior, 4)
        raw: Dict[Action, float] = {}
        vals: List[float] = []
        for i in sorted(INFO_MODE_TABLE.keys()):
            for c in sorted(COT_MODE_TABLE.keys()):
                v = math.log(max(1e-9, float(ip[i - 1]))) + math.log(max(1e-9, float(cp[c])))
                raw[(i, c)] = float(v)
                vals.append(float(v))
        mean_v = sum(vals) / max(1, len(vals))
        return {k: float(v - mean_v) for k, v in raw.items()}

    def diversity_regularization_loss(
        self,
        *,
        margin: Optional[float] = None,
        slot_names: Optional[Sequence[str]] = None,
    ) -> float:
        names = list(slot_names) if slot_names is not None else sorted(self.slots.keys())
        if len(names) <= 1:
            return 0.0
        m = float(self.diversity_margin if margin is None else margin)
        vals: List[float] = []
        for i in range(len(names)):
            key_i = self.slots[names[i]].get("slot_key", [])
            for j in range(i + 1, len(names)):
                key_j = self.slots[names[j]].get("slot_key", [])
                cos = self._cosine(key_i, key_j)
                vals.append(max(0.0, cos - m) ** 2)
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def _max_cosine_pair(self) -> Tuple[Optional[Tuple[str, str]], float]:
        names = sorted(self.slots.keys())
        best_pair: Optional[Tuple[str, str]] = None
        best_cos = -1.0
        for i in range(len(names)):
            ki = self.slots[names[i]].get("slot_key", [])
            for j in range(i + 1, len(names)):
                kj = self.slots[names[j]].get("slot_key", [])
                cos = self._cosine(ki, kj)
                if cos > best_cos:
                    best_cos = float(cos)
                    best_pair = (names[i], names[j])
        return best_pair, float(best_cos)

    def _merge_slots(self, keep_name: str, drop_name: str, cosine: float) -> Dict[str, object]:
        a = self.slots[keep_name]
        b = self.slots[drop_name]
        ca = max(1.0, float(a.get("support_count", a.get("count", 0.0) or 0.0) or 1.0))
        cb = max(1.0, float(b.get("support_count", b.get("count", 0.0) or 0.0) or 1.0))
        weight_sum = ca + cb

        key_a = self._normalize_vec(a.get("slot_key", []), self.encoder_dim)
        key_b = self._normalize_vec(b.get("slot_key", []), self.encoder_dim)
        merged_key = self._normalize_vec(
            [(ca * va + cb * vb) / weight_sum for va, vb in zip(key_a, key_b)],
            self.encoder_dim,
        )
        ip_a = self._normalize_prob(a.get("info_prior", []), 5)
        ip_b = self._normalize_prob(b.get("info_prior", []), 5)
        cp_a = self._normalize_prob(a.get("cot_prior", []), 4)
        cp_b = self._normalize_prob(b.get("cot_prior", []), 4)
        merged_info = self._normalize_prob([(ca * x + cb * y) / weight_sum for x, y in zip(ip_a, ip_b)], 5)
        merged_cot = self._normalize_prob([(ca * x + cb * y) / weight_sum for x, y in zip(cp_a, cp_b)], 4)
        reward_a = float(a.get("avg_reward", 0.0) or 0.0)
        reward_b = float(b.get("avg_reward", 0.0) or 0.0)

        merged = {
            "slot_key": merged_key,
            "info_prior": merged_info,
            "cot_prior": merged_cot,
            "count": float(a.get("count", 0.0) or 0.0) + float(b.get("count", 0.0) or 0.0),
            "avg_reward": float((ca * reward_a + cb * reward_b) / weight_sum),
            "support_count": float(ca + cb),
            "merged_from": sorted(set([keep_name, drop_name] + list(a.get("merged_from", [])) + list(b.get("merged_from", [])))),
        }
        self.slots[keep_name] = merged
        self.reserve_slots[drop_name] = b
        del self.slots[drop_name]
        return {"keep": keep_name, "drop": drop_name, "cosine": float(cosine)}

    def separation_check_and_merge(
        self,
        *,
        target_slot_count: Optional[int] = None,
        cosine_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        target = max(1, int(self.target_slot_count if target_slot_count is None else target_slot_count))
        threshold = float(self.merge_cosine_threshold if cosine_threshold is None else cosine_threshold)
        initial_count = len(self.slots)
        merges: List[Dict[str, object]] = []

        # Pass-1: merge highly overlapped slots.
        while len(self.slots) > 1:
            pair, best_cos = self._max_cosine_pair()
            if pair is None or best_cos < threshold:
                break
            merges.append(self._merge_slots(pair[0], pair[1], best_cos))

        # Pass-2: if still above target, continue merging most similar pairs.
        while len(self.slots) > target and len(self.slots) > 1:
            pair, best_cos = self._max_cosine_pair()
            if pair is None:
                break
            merges.append(self._merge_slots(pair[0], pair[1], best_cos))

        report = {
            "initial_slot_count": int(initial_count),
            "final_slot_count": int(len(self.slots)),
            "target_slot_count": int(target),
            "cosine_threshold": float(threshold),
            "merges": merges,
        }
        self._last_merge_report = report
        return report

    def _refill_from_reserve(self, *, target_slot_count: int) -> int:
        target = max(1, int(target_slot_count))
        if len(self.slots) >= target or not self.reserve_slots:
            return 0
        ranked = sorted(
            self.reserve_slots.items(),
            key=lambda kv: (
                float(kv[1].get("support_count", kv[1].get("count", 0.0) or 0.0) or 0.0),
                float(kv[1].get("avg_reward", 0.0) or 0.0),
                kv[0],
            ),
            reverse=True,
        )
        added = 0
        for name, payload in ranked:
            if len(self.slots) >= target:
                break
            if name in self.slots:
                continue
            self.slots[name] = payload
            del self.reserve_slots[name]
            added += 1
        return added

    def _apply_diversity_push(self, *, steps: int = 6) -> Dict[str, float]:
        before = self.diversity_regularization_loss(margin=self.diversity_margin)
        if (steps <= 0) or (len(self.slots) <= 1) or (self.diversity_weight <= 0.0):
            stats = {"before": float(before), "after": float(before)}
            self._last_diversity_loss_stats = stats
            return stats
        step_size = 0.04 * min(1.0, float(self.diversity_weight))
        names = sorted(self.slots.keys())
        for _ in range(int(steps)):
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ni = names[i]
                    nj = names[j]
                    ki = self._normalize_vec(self.slots[ni].get("slot_key", []), self.encoder_dim)
                    kj = self._normalize_vec(self.slots[nj].get("slot_key", []), self.encoder_dim)
                    cos = self._cosine(ki, kj)
                    if cos <= float(self.diversity_margin):
                        continue
                    diff = [float(a - b) for a, b in zip(ki, kj)]
                    norm = math.sqrt(sum(v * v for v in diff))
                    if norm <= 1e-12:
                        idx = int(hashlib.md5(f"{ni}|{nj}".encode("utf-8")).digest()[0]) % self.encoder_dim
                        diff[idx] = 1.0
                        norm = 1.0
                    unit = [v / norm for v in diff]
                    ki_new = self._normalize_vec([a + (step_size * u) for a, u in zip(ki, unit)], self.encoder_dim)
                    kj_new = self._normalize_vec([b - (step_size * u) for b, u in zip(kj, unit)], self.encoder_dim)
                    self.slots[ni]["slot_key"] = ki_new
                    self.slots[nj]["slot_key"] = kj_new
        after = self.diversity_regularization_loss(margin=self.diversity_margin)
        stats = {"before": float(before), "after": float(after)}
        self._last_diversity_loss_stats = stats
        return stats

    def fit_match_temperature(
        self,
        *,
        samples: Optional[List[Dict[str, object]]] = None,
        context_tag: str = "non_code",
    ) -> Dict[str, float]:
        if not bool(self.match_temperature_learnable):
            report = {"before": float(self.match_temperature), "after": float(self.match_temperature), "sample_count": 0.0}
            self._last_temperature_report = report
            return report
        names = sorted(self.slots.keys())
        if len(names) <= 1:
            report = {"before": float(self.match_temperature), "after": float(self.match_temperature), "sample_count": 0.0}
            self._last_temperature_report = report
            return report

        if not samples:
            samples = []
            for n in names:
                samples.append({"query": self.slots[n].get("slot_key", []), "target_slot": n, "context_tag": context_tag})
        valid = []
        for sample in samples:
            query = sample.get("query")
            target_slot = str(sample.get("target_slot", "") or "")
            if not isinstance(query, (list, tuple)) or target_slot not in self.slots:
                continue
            valid.append({"query": [float(x) for x in query], "target_slot": target_slot, "context_tag": str(sample.get("context_tag", context_tag))})
        if not valid:
            report = {"before": float(self.match_temperature), "after": float(self.match_temperature), "sample_count": 0.0}
            self._last_temperature_report = report
            return report

        def _objective(tau: float) -> Tuple[float, float]:
            ent_vals: List[float] = []
            nll_vals: List[float] = []
            for sample in valid:
                ctag = str(sample.get("context_tag", context_tag))
                sims = []
                for n in names:
                    cos = self._cosine(sample["query"], self.slots[n].get("slot_key", []))
                    sims.append(cos + self._context_bias(n, ctag))
                probs = self._softmax(sims, temperature=float(tau))
                entropy = -sum(float(p) * math.log(max(1e-9, float(p))) for p in probs)
                ent_vals.append(entropy)
                tgt = str(sample["target_slot"])
                tgt_idx = names.index(tgt)
                nll_vals.append(-math.log(max(1e-9, float(probs[tgt_idx]))))
            mean_ent = float(sum(ent_vals) / len(ent_vals))
            mean_nll = float(sum(nll_vals) / len(nll_vals))
            target_ent = 0.45 * math.log(max(2.0, float(len(names))))
            obj = mean_nll + (0.12 * abs(mean_ent - target_ent))
            return float(obj), float(mean_ent)

        best_tau = float(self.match_temperature)
        best_obj = float("inf")
        best_entropy = 0.0
        tau_grid = [0.35 + (0.05 * i) for i in range(34)]
        for tau in tau_grid:
            obj, mean_ent = _objective(tau)
            if obj < best_obj:
                best_obj = obj
                best_tau = float(tau)
                best_entropy = float(mean_ent)
        before = float(self.match_temperature)
        self.match_temperature = float(best_tau)
        report = {
            "before": float(before),
            "after": float(self.match_temperature),
            "sample_count": float(len(valid)),
            "objective": float(best_obj),
            "mean_entropy": float(best_entropy),
        }
        self._last_temperature_report = report
        return report

    def update_slot_stats(
        self,
        *,
        task_query: Sequence[float],
        reward: float,
        info_mode: int,
        cot_mode: int,
        alpha: float = 0.08,
        context_tag: str = "non_code",
    ) -> None:
        if not self.slots:
            return
        retrieval = self.retrieve(task_query=task_query, context_tag=context_tag)
        names = [str(x) for x in retrieval.get("slot_names", [])]
        weights = [float(x) for x in retrieval.get("slot_weights", [])]
        for w, name in zip(weights, names):
            slot = self.slots.get(name)
            if slot is None:
                continue
            prev_count = float(slot.get("count", 0.0) or 0.0)
            prev_avg = float(slot.get("avg_reward", 0.0) or 0.0)
            slot["count"] = prev_count + float(w)
            slot["support_count"] = float(slot.get("support_count", 0.0) or 0.0) + float(w)
            slot["avg_reward"] = (1.0 - alpha) * prev_avg + alpha * float(w) * float(reward)

            ip = self._normalize_prob(slot.get("info_prior", []), 5)
            cp = self._normalize_prob(slot.get("cot_prior", []), 4)
            i = max(1, min(5, int(info_mode))) - 1
            c = max(0, min(3, int(cot_mode)))
            ip[i] = (1.0 - alpha) * ip[i] + alpha * max(1e-6, 1.0 + float(reward))
            cp[c] = (1.0 - alpha) * cp[c] + alpha * max(1e-6, 1.0 + float(reward))
            slot["info_prior"] = self._normalize_prob(ip, 5)
            slot["cot_prior"] = self._normalize_prob(cp, 4)

    def initialize_from_rows(self, *, task_encoder: TaskEncoder, rows: List[Dict[str, object]]) -> int:
        default_slots = self._default_slots(self.encoder_dim)
        stats: Dict[str, Dict[str, object]] = {}
        for slot_name in TaskEncoder.SLOT_CANDIDATES:
            stats[slot_name] = {
                "support": 0.0,
                "sum_reward": 0.0,
                "sum_query": [0.0] * self.encoder_dim,
                "info_hist": [0.0] * 5,
                "cot_hist": [0.0] * 4,
            }
        temp_samples: List[Dict[str, object]] = []
        used = 0
        for row in rows:
            try:
                prompt = self._extract_prompt(row)
                if not prompt:
                    continue
                info_mode, cot_mode = self._extract_modes(row)
                reward = self._extract_reward(row)
            except Exception:
                continue
            query = task_encoder.encode(prompt)
            slot_scores = task_encoder.slot_signal(prompt)
            best_slot = max(slot_scores.items(), key=lambda kv: (float(kv[1]), kv[0]))[0]
            s = stats[best_slot]
            s["support"] = float(s["support"]) + 1.0
            s["sum_reward"] = float(s["sum_reward"]) + float(reward)
            s["sum_query"] = [float(x) + float(y) for x, y in zip(s["sum_query"], query)]
            s["info_hist"][info_mode - 1] = float(s["info_hist"][info_mode - 1]) + 1.0
            s["cot_hist"][cot_mode] = float(s["cot_hist"][cot_mode]) + 1.0
            temp_samples.append({"query": query, "target_slot": best_slot, "context_tag": "code" if str(row.get("category", "")).strip().lower() == "code" else "non_code"})
            used += 1

        candidates: Dict[str, Dict[str, object]] = {}
        for slot_name in TaskEncoder.SLOT_CANDIDATES:
            s = stats[slot_name]
            support = float(s["support"])
            if support > 0.0:
                mean_query = [float(v) / support for v in s["sum_query"]]
                key = self._normalize_vec(mean_query, self.encoder_dim)
                tpl_info, tpl_cot = self._slot_prior_template(slot_name)
                obs_info = self._normalize_prob(s["info_hist"], 5)
                obs_cot = self._normalize_prob(s["cot_hist"], 4)
                info_prior = self._normalize_prob([(0.70 * o) + (0.30 * t) for o, t in zip(obs_info, tpl_info)], 5)
                cot_prior = self._normalize_prob([(0.70 * o) + (0.30 * t) for o, t in zip(obs_cot, tpl_cot)], 4)
                avg_reward = float(s["sum_reward"]) / support
            else:
                seed = default_slots[slot_name]
                key = self._normalize_vec(seed.get("slot_key", []), self.encoder_dim)
                info_prior = self._normalize_prob(seed.get("info_prior", []), 5)
                cot_prior = self._normalize_prob(seed.get("cot_prior", []), 4)
                avg_reward = 0.0
            candidates[slot_name] = {
                "slot_key": key,
                "info_prior": info_prior,
                "cot_prior": cot_prior,
                "count": float(support),
                "support_count": float(support),
                "avg_reward": float(avg_reward),
            }

        ranked = sorted(
            candidates.keys(),
            key=lambda name: (
                float(candidates[name].get("support_count", 0.0) or 0.0),
                float(candidates[name].get("avg_reward", 0.0) or 0.0),
                name,
            ),
            reverse=True,
        )
        selected = [n for n in ranked if float(candidates[n].get("support_count", 0.0) or 0.0) > 0.0]
        if not selected:
            selected = ranked[: self.target_slot_count]
        if len(selected) < self.target_slot_count:
            for n in ranked:
                if n in selected:
                    continue
                selected.append(n)
                if len(selected) >= self.target_slot_count:
                    break

        self.slots = {name: candidates[name] for name in selected}
        self.reserve_slots = {name: candidates[name] for name in ranked if name not in self.slots}
        merge_report = self.separation_check_and_merge(
            target_slot_count=int(self.target_slot_count),
            cosine_threshold=float(self.merge_cosine_threshold),
        )
        refill_added = self._refill_from_reserve(target_slot_count=int(self.target_slot_count))
        if len(self.slots) > int(self.target_slot_count):
            self.separation_check_and_merge(
                target_slot_count=int(self.target_slot_count),
                cosine_threshold=-1.0,
            )
        if len(self.slots) < int(self.target_slot_count):
            self._refill_from_reserve(target_slot_count=int(self.target_slot_count))
        diversity_stats = self._apply_diversity_push(steps=8)
        temp_report = self.fit_match_temperature(samples=temp_samples)
        merge_report["refill_added"] = int(refill_added)
        merge_report["final_slot_count"] = int(len(self.slots))
        self._last_merge_report = merge_report
        self._last_diversity_loss_stats = diversity_stats
        self._last_temperature_report = temp_report
        return int(used)

    def to_dict(self) -> Dict[str, object]:
        return {
            "encoder_dim": int(self.encoder_dim),
            "slots": self.slots,
            "reserve_slots": self.reserve_slots,
            "target_slot_count": int(self.target_slot_count),
            "merge_cosine_threshold": float(self.merge_cosine_threshold),
            "diversity_weight": float(self.diversity_weight),
            "diversity_margin": float(self.diversity_margin),
            "match_temperature": float(self.match_temperature),
            "match_temperature_learnable": bool(self.match_temperature_learnable),
            "slot_schema_version": str(self.slot_schema_version),
            "slot_merge_report": self._last_merge_report,
            "diversity_loss_stats": self._last_diversity_loss_stats,
            "temperature_report": self._last_temperature_report,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SlotMemory":
        slots = payload.get("slots")
        reserve_slots = payload.get("reserve_slots")
        if not isinstance(slots, dict):
            slots = None
        if not isinstance(reserve_slots, dict):
            reserve_slots = None
        obj = cls(
            encoder_dim=int(payload.get("encoder_dim", 24) or 24),
            slots=slots,
            reserve_slots=reserve_slots,
            target_slot_count=int(payload.get("target_slot_count", 15) or 15),
            merge_cosine_threshold=float(payload.get("merge_cosine_threshold", 0.90) or 0.90),
            diversity_weight=float(payload.get("diversity_weight", 0.02) or 0.02),
            diversity_margin=float(payload.get("diversity_margin", 0.35) or 0.35),
            match_temperature_init=float(payload.get("match_temperature", 1.0) or 1.0),
            match_temperature_learnable=bool(payload.get("match_temperature_learnable", True)),
            slot_schema_version=str(payload.get("slot_schema_version", "v12_15slot")),
        )
        slot_merge_report = payload.get("slot_merge_report")
        if isinstance(slot_merge_report, dict):
            obj._last_merge_report = slot_merge_report
        diversity_loss_stats = payload.get("diversity_loss_stats")
        if isinstance(diversity_loss_stats, dict):
            obj._last_diversity_loss_stats = {
                "before": float(diversity_loss_stats.get("before", 0.0) or 0.0),
                "after": float(diversity_loss_stats.get("after", 0.0) or 0.0),
            }
        temperature_report = payload.get("temperature_report")
        if isinstance(temperature_report, dict):
            obj._last_temperature_report = {
                "before": float(temperature_report.get("before", obj.match_temperature) or obj.match_temperature),
                "after": float(temperature_report.get("after", obj.match_temperature) or obj.match_temperature),
                "sample_count": float(temperature_report.get("sample_count", 0.0) or 0.0),
            }
        return obj


class GovernanceHead:
    def __init__(self, *, scale: float = 1.0) -> None:
        self.scale = float(scale)
        self.encoder = GovernanceFeatureEncoder.default()
        self.model = None
        self.model_path: str = ""
        self.metadata: Dict[str, object] = {}

    @property
    def has_model(self) -> bool:
        return self.model is not None

    @property
    def feature_schema_version(self) -> str:
        return str(self.metadata.get("feature_schema_version", self.encoder.feature_schema_version))

    @property
    def hidden_proj_dim(self) -> int:
        return int(self.metadata.get("hidden_proj_dim", self.encoder.hidden_proj_dim))

    @property
    def use_hidden_observables(self) -> bool:
        return bool(self.metadata.get("use_hidden_observables", self.encoder.use_hidden_observables))

    def load_bundle(self, path: str | Path) -> bool:
        src = Path(path).expanduser().resolve()
        if not src.exists():
            return False
        bundle = load_governance_bundle(src)
        self.encoder = bundle["encoder"]
        self.model = bundle["model"]
        self.model_path = str(bundle["path"])
        self.metadata = dict(bundle.get("metadata", {}))
        self.metadata.setdefault("feature_schema_version", self.encoder.feature_schema_version)
        self.metadata.setdefault("hidden_proj_dim", int(self.encoder.hidden_proj_dim))
        self.metadata.setdefault("use_hidden_observables", bool(self.encoder.use_hidden_observables))
        return True

    def encode_state(
        self,
        *,
        context_tag: str,
        entropy: float,
        margin: float,
        prompt_len: int,
        generated_tokens: int,
        progress_ratio: float,
        hidden_l2_norm: float = 0.0,
        top1_prob: float = 0.0,
        top2_prob: float = 0.0,
        topk_mass: float = 0.0,
        eos_prob: float = 0.0,
        eos_rank: float = 0.0,
        repeat_ngram_ratio: float = 0.0,
        current_info_mode: int = 0,
        current_cot_mode: int = 0,
        remaining_budget_ratio: float = 1.0,
        segment_progress_ratio: float = 0.0,
        is_answer_zone: bool = False,
        is_code_mode: bool = False,
        boundary_kind: str = "none",
        hidden_state_proj: Sequence[float] | None = None,
    ) -> List[float]:
        vec = self.encoder.encode_state(
            context_tag=context_tag,
            entropy=entropy,
            margin=margin,
            prompt_len=prompt_len,
            generated_tokens=generated_tokens,
            progress_ratio=progress_ratio,
            hidden_l2_norm=hidden_l2_norm,
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            topk_mass=topk_mass,
            eos_prob=eos_prob,
            eos_rank=eos_rank,
            repeat_ngram_ratio=repeat_ngram_ratio,
            current_info_mode=current_info_mode,
            current_cot_mode=current_cot_mode,
            remaining_budget_ratio=remaining_budget_ratio,
            segment_progress_ratio=segment_progress_ratio,
            is_answer_zone=is_answer_zone,
            is_code_mode=is_code_mode,
            boundary_kind=boundary_kind,
            hidden_state_proj=hidden_state_proj,
        )
        return vec.tolist()

    def action_scores(self, feat: List[float]) -> Dict[Action, float]:
        if self.model is None:
            return {}
        out = self.model.forward(feat)
        scores: Dict[Action, float] = {}
        for info_mode in sorted(INFO_MODE_TABLE.keys()):
            for cot_mode in sorted(COT_MODE_TABLE.keys()):
                scores[(info_mode, cot_mode)] = float(self.scale) * float(
                    out["info_log_probs"][int(info_mode) - 1] + out["cot_log_probs"][int(cot_mode)]
                )
        return scores

    def dump(self) -> Dict[str, object]:
        return {
            "model_path": self.model_path,
            "has_model": self.has_model,
            "scale": self.scale,
            "feature_dim": self.encoder.feature_dim,
            "feature_schema_version": self.feature_schema_version,
            "hidden_proj_dim": self.hidden_proj_dim,
            "use_hidden_observables": self.use_hidden_observables,
        }

    def load(self, payload: Dict[str, object]) -> None:
        path = str(payload.get("model_path", "")).strip()
        if path:
            self.load_bundle(path)


class AdaptiveModeController:
    def __init__(self, cfg: Optional[AdaptiveModeControllerConfig] = None) -> None:
        self.cfg = cfg or AdaptiveModeControllerConfig()
        self._entropy_stats = EmaStats()
        self._margin_stats = EmaStats()
        self._plen_stats = EmaStats()
        self._gen_tok_stats = EmaStats()
        self._last_action: Optional[Action] = None
        self._reward_ema: Dict[Action, float] = {}
        self._reward_ema_by_model: Dict[str, Dict[Action, float]] = {}
        self._context_bank: Dict[str, Dict[Action, float]] = {}
        self._episode_bank: Dict[str, Dict[Action, Deque[float]]] = {}
        self._model_tag: str = "default"
        self._episode_traj: List[Action] = []
        self._step_idx = 0
        self._last_switch_step = -10**9
        self._switch_hist: Deque[int] = deque(maxlen=max(4, int(self.cfg.recent_switch_window)))
        self._guardrail_masks_applied = 0
        self._head = GovernanceHead(scale=float(self.cfg.head_scale))
        if str(self.cfg.governance_model_path).strip():
            self._head.load_bundle(self.cfg.governance_model_path)

        # proposal-aligned task-conditioned memory components
        self._task_encoder = TaskEncoder(
            dim=int(self.cfg.task_encoder_dim),
            slot_schema_version=str(self.cfg.slot_schema_version),
        )
        self._slot_memory = SlotMemory(
            encoder_dim=int(self.cfg.task_encoder_dim),
            target_slot_count=int(self.cfg.target_slot_count),
            merge_cosine_threshold=float(self.cfg.slot_merge_cosine_threshold),
            diversity_weight=float(self.cfg.slot_diversity_weight),
            diversity_margin=float(self.cfg.slot_diversity_margin),
            match_temperature_init=float(self.cfg.slot_match_temperature_init),
            match_temperature_learnable=bool(self.cfg.slot_match_temperature_learnable),
            slot_schema_version=str(self.cfg.slot_schema_version),
        )
        if self._head.metadata:
            self._load_policy_components_from_head_metadata()

    def _load_policy_components_from_head_metadata(self) -> None:
        md = self._head.metadata if isinstance(self._head.metadata, dict) else {}
        te = md.get("task_encoder")
        sm = md.get("slot_memory")
        if isinstance(te, dict):
            self._task_encoder = TaskEncoder.from_dict(te)
            self.cfg.task_encoder_dim = int(self._task_encoder.dim)
        if isinstance(sm, dict):
            self._slot_memory = SlotMemory.from_dict(sm)
            self.cfg.target_slot_count = int(self._slot_memory.target_slot_count)
            self.cfg.slot_merge_cosine_threshold = float(self._slot_memory.merge_cosine_threshold)
            self.cfg.slot_diversity_weight = float(self._slot_memory.diversity_weight)
            self.cfg.slot_diversity_margin = float(self._slot_memory.diversity_margin)
            self.cfg.slot_match_temperature_init = float(self._slot_memory.match_temperature)
            self.cfg.slot_match_temperature_learnable = bool(self._slot_memory.match_temperature_learnable)
            self.cfg.slot_schema_version = str(self._slot_memory.slot_schema_version)

    def apply_runtime_policy_config(self, cfg: AdaptiveModeControllerConfig) -> None:
        """Re-apply runtime policy knobs from config after loading model metadata/state.

        Governance bundle metadata may carry training-time slot memory payloads.
        For ablations and strict proposal runs, runtime config must take precedence.
        """
        self.cfg.memory_type = str(cfg.memory_type)
        self.cfg.task_encoder_enabled = bool(cfg.task_encoder_enabled)
        self.cfg.slot_prior_weight = float(cfg.slot_prior_weight)
        self.cfg.task_encoder_dim = int(cfg.task_encoder_dim)
        self.cfg.target_slot_count = int(cfg.target_slot_count)
        self.cfg.slot_merge_cosine_threshold = float(cfg.slot_merge_cosine_threshold)
        self.cfg.slot_diversity_weight = float(cfg.slot_diversity_weight)
        self.cfg.slot_diversity_margin = float(cfg.slot_diversity_margin)
        self.cfg.slot_match_temperature_init = float(cfg.slot_match_temperature_init)
        self.cfg.slot_match_temperature_learnable = bool(cfg.slot_match_temperature_learnable)
        self.cfg.slot_schema_version = str(cfg.slot_schema_version)
        self.cfg.use_hidden_observables = bool(cfg.use_hidden_observables)
        self.cfg.online_memory_update_enabled = bool(cfg.online_memory_update_enabled)

        if (
            int(self._task_encoder.dim) != int(self.cfg.task_encoder_dim)
            or str(self._task_encoder.slot_schema_version) != str(self.cfg.slot_schema_version)
        ):
            self._task_encoder = TaskEncoder(
                dim=int(self.cfg.task_encoder_dim),
                slot_schema_version=str(self.cfg.slot_schema_version),
            )

        self._slot_memory.target_slot_count = int(self.cfg.target_slot_count)
        self._slot_memory.merge_cosine_threshold = float(self.cfg.slot_merge_cosine_threshold)
        self._slot_memory.diversity_weight = float(self.cfg.slot_diversity_weight)
        self._slot_memory.diversity_margin = float(self.cfg.slot_diversity_margin)
        self._slot_memory.match_temperature = max(0.05, float(self.cfg.slot_match_temperature_init))
        self._slot_memory.match_temperature_learnable = bool(self.cfg.slot_match_temperature_learnable)
        self._slot_memory.slot_schema_version = str(self.cfg.slot_schema_version)

    def _action_space(self) -> List[Action]:
        cot_levels = sorted(COT_MODE_TABLE.keys())
        if not bool(self.cfg.use_cot0):
            cot_levels = [c for c in cot_levels if c > 0]
        return [(i, c) for i in sorted(INFO_MODE_TABLE.keys()) for c in cot_levels]

    @property
    def guardrail_masks_applied(self) -> int:
        return int(self._guardrail_masks_applied)

    @property
    def hidden_proj_dim(self) -> int:
        if not bool(self.cfg.use_hidden_observables):
            return 0
        return int(self._head.hidden_proj_dim or self.cfg.hidden_proj_dim)

    @property
    def governance_model_loaded(self) -> bool:
        return self._head.has_model

    @property
    def governance_model_metadata(self) -> Dict[str, object]:
        return {
            "path": self._head.model_path,
            "loaded": self._head.has_model,
            "feature_schema_version": self._head.feature_schema_version,
            "hidden_proj_dim": self._head.hidden_proj_dim,
            "feature_dim": self._head.encoder.feature_dim,
            "use_hidden_observables": bool(self._head.use_hidden_observables),
            "online_memory_update_enabled": bool(self.cfg.online_memory_update_enabled),
            "memory_type": str(self.cfg.memory_type),
            "task_encoder_enabled": bool(self.cfg.task_encoder_enabled),
            "slot_prior_weight": float(self.cfg.slot_prior_weight),
            "slot_schema_version": str(self._slot_memory.slot_schema_version),
            "slot_count": int(len(self._slot_memory.slots)),
            "final_slot_names": self._slot_memory.final_slot_names,
            "slot_merge_report": self._slot_memory.last_merge_report,
            "match_temperature": float(self._slot_memory.match_temperature),
            "diversity_loss_stats": self._slot_memory.last_diversity_loss_stats,
        }

    def begin_episode(self) -> None:
        self._episode_traj = []
        self._last_action = None
        self._step_idx = 0
        self._last_switch_step = -10**9
        self._switch_hist.clear()
        self._guardrail_masks_applied = 0

    def load_governance_model(self, path: str | Path) -> bool:
        ok = self._head.load_bundle(path)
        if ok:
            self._load_policy_components_from_head_metadata()
        return ok

    def set_model_tag(self, model_tag: str) -> None:
        tag = str(model_tag).strip() or "default"
        self._model_tag = tag
        self._reward_ema_by_model.setdefault(tag, {})

    def _normalize_context_tag(self, context_tag: Optional[str]) -> str:
        tag = str(context_tag or "").strip().lower()
        if tag in {"code", "non_code"}:
            return tag
        if not tag:
            return "default"
        # Keep memory policy generic; disallow dataset/benchmark-specific context buckets.
        return "non_code"

    def _memory_weights(self) -> Tuple[float, float, float]:
        wg = max(0.0, float(self.cfg.global_memory_weight))
        wm = max(0.0, float(self.cfg.model_memory_weight))
        we = max(0.0, float(self.cfg.episode_memory_weight))
        if not bool(self.cfg.enable_model_memory):
            wm = 0.0
        total = wg + wm + we
        if total <= 1e-12:
            return 1.0, 0.0, 0.0
        return wg / total, wm / total, we / total

    def _episode_context_value(self, *, context_tag: str, action: Action, fallback: float) -> float:
        bucket = self._episode_bank.get(context_tag, {})
        vals = bucket.get(action)
        if vals:
            return float(sum(vals) / max(1, len(vals)))
        ctx = self._context_bank.get(context_tag, {})
        if action in ctx:
            return float(ctx[action])
        return float(fallback)

    def _uncertainty_score(self, *, entropy: float, margin: float, prompt_len: int) -> Dict[str, float]:
        z_e = self._entropy_stats.zscore(entropy)
        z_m = self._margin_stats.zscore(margin)
        z_l = self._plen_stats.zscore(float(prompt_len))
        raw = z_e - z_m + 0.1 * z_l
        hard_prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, raw))))
        return {
            "uncertainty_raw": raw,
            "uncertainty_hard_prob": hard_prob,
            "entropy_z": z_e,
            "margin_z": z_m,
            "prompt_len_z": z_l,
        }

    def _legacy_memory_score(self, *, action: Action, context_tag: str) -> float:
        if float(self.cfg.score_memory_weight) <= 0.0:
            return 0.0
        model_mem = self._reward_ema_by_model.get(self._model_tag, {})
        wg, wm, we = self._memory_weights()
        g_mem = float(self._reward_ema.get(action, 0.0))
        m_mem = float(model_mem.get(action, g_mem))
        e_mem = self._episode_context_value(context_tag=context_tag, action=action, fallback=g_mem)
        return float(self.cfg.score_memory_weight) * ((wg * g_mem) + (wm * m_mem) + (we * e_mem))

    def _slot_prior_payload(self, *, prompt_text: str, context_tag: str) -> Dict[str, object]:
        if not bool(self.cfg.task_encoder_enabled):
            return {"info_prior": [0.2] * 5, "cot_prior": [0.25] * 4, "slot_names": [], "slot_weights": []}
        query = self._task_encoder.encode(prompt_text)
        out = self._slot_memory.retrieve(task_query=query, context_tag=context_tag)
        out["task_query"] = query
        return out

    def _apply_guardrails(
        self,
        *,
        actions: Dict[Action, float],
        remaining_budget_ratio: float,
        is_answer_zone: bool,
        current_cot_mode: int,
    ) -> Dict[Action, float]:
        filtered = dict(actions)
        initial_count = len(filtered)
        if remaining_budget_ratio <= 1e-6:
            allow_cot = 0 if bool(self.cfg.use_cot0) else min(1, max(1, int(current_cot_mode or 1)))
            filtered = {action: score for action, score in filtered.items() if int(action[1]) <= int(allow_cot)}
        if is_answer_zone:
            filtered = {
                action: score
                for action, score in filtered.items()
                if int(action[0]) <= int(self.cfg.answer_zone_max_info_mode)
                and int(action[1]) <= int(self.cfg.answer_zone_max_cot_mode)
            }
        if not filtered:
            filtered = dict(actions)
        self._guardrail_masks_applied += max(0, initial_count - len(filtered))
        return filtered

    def choose(
        self,
        *,
        entropy: float,
        margin: float,
        prompt_len: int,
        generated_tokens: int = 0,
        progress_ratio: float = 0.0,
        context_tag: Optional[str] = None,
        hidden_l2_norm: float = 0.0,
        top1_prob: float = 0.0,
        top2_prob: float = 0.0,
        topk_mass: float = 0.0,
        eos_prob: float = 0.0,
        eos_rank: float = 0.0,
        repeat_ngram_ratio: float = 0.0,
        current_info_mode: int = 0,
        current_cot_mode: int = 0,
        remaining_budget_ratio: float = 1.0,
        segment_progress_ratio: float = 0.0,
        is_answer_zone: bool = False,
        is_code_mode: bool = False,
        boundary_kind: str = "none",
        hidden_state_proj: Sequence[float] | None = None,
        prompt_text: str = "",
    ) -> Tuple[int, int, Dict[str, float]]:
        feats = self._uncertainty_score(entropy=entropy, margin=margin, prompt_len=prompt_len)
        feats["head_model_loaded"] = 1.0 if self._head.has_model else 0.0
        self._step_idx += 1
        ctx_tag = self._normalize_context_tag(context_tag)

        # strict proposal mode: avoid hidden-state dependence in governance inputs
        if not bool(self.cfg.use_hidden_observables):
            hidden_l2_norm = 0.0
            hidden_state_proj = []

        slot_payload = self._slot_prior_payload(prompt_text=prompt_text, context_tag=ctx_tag)
        info_prior = [float(x) for x in slot_payload.get("info_prior", [0.2] * 5)]
        cot_prior = [float(x) for x in slot_payload.get("cot_prior", [0.25] * 4)]
        slot_bonus = self._slot_memory.action_bonus(info_prior=info_prior, cot_prior=cot_prior)

        head_feat = self._head.encode_state(
            context_tag=ctx_tag,
            entropy=entropy,
            margin=margin,
            prompt_len=prompt_len,
            generated_tokens=generated_tokens,
            progress_ratio=progress_ratio,
            hidden_l2_norm=hidden_l2_norm,
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            topk_mass=topk_mass,
            eos_prob=eos_prob,
            eos_rank=eos_rank,
            repeat_ngram_ratio=repeat_ngram_ratio,
            current_info_mode=current_info_mode,
            current_cot_mode=current_cot_mode,
            remaining_budget_ratio=remaining_budget_ratio,
            segment_progress_ratio=segment_progress_ratio,
            is_answer_zone=is_answer_zone,
            is_code_mode=is_code_mode,
            boundary_kind=boundary_kind,
            hidden_state_proj=hidden_state_proj,
        )
        base_scores = {
            action: score
            for action, score in self._head.action_scores(head_feat).items()
            if action in self._action_space()
        }
        if not base_scores:
            base_scores = {action: 0.0 for action in self._action_space()}
        scores = self._apply_guardrails(
            actions=base_scores,
            remaining_budget_ratio=remaining_budget_ratio,
            is_answer_zone=is_answer_zone,
            current_cot_mode=current_cot_mode,
        )

        for action in list(scores):
            s = float(scores[action])
            s += self._legacy_memory_score(action=action, context_tag=ctx_tag)
            s += float(self.cfg.slot_prior_weight) * float(slot_bonus.get(action, 0.0))
            scores[action] = s

        best_action = max(scores.items(), key=lambda x: x[1])[0]
        switched = False
        best_score = float(scores.get(best_action, -1e9))
        if self._last_action is not None and best_action != self._last_action:
            curr_score = float(scores.get(self._last_action, base_scores.get(self._last_action, -1e9)))
            min_gap = max(0, int(self.cfg.min_steps_between_switch))
            if (self._step_idx - self._last_switch_step) < min_gap:
                best_action = self._last_action
            switch_margin = float(self.cfg.switch_margin)
            if bool(self.cfg.dynamic_hysteresis):
                recent = float(sum(self._switch_hist)) / max(1.0, float(len(self._switch_hist) or 1))
                switch_margin += float(self.cfg.hysteresis_base) + (
                    float(self.cfg.hysteresis_switch_scale) * recent
                )
            if best_score <= curr_score + switch_margin:
                best_action = self._last_action
            else:
                switched = True

        self._last_action = best_action
        if switched:
            self._last_switch_step = self._step_idx
            self._switch_hist.append(1)
        else:
            self._switch_hist.append(0)
        self._episode_traj.append(best_action)

        slot_names = [str(x) for x in slot_payload.get("slot_names", [])]
        slot_weights = [float(x) for x in slot_payload.get("slot_weights", [])]
        for name, w in zip(slot_names, slot_weights):
            feats[f"slot_weight_{name}"] = float(w)
        return int(best_action[0]), int(best_action[1]), feats

    def _update_action_reward(
        self,
        action: Action,
        reward: float,
        alpha: float = 0.1,
        context_tag: Optional[str] = None,
    ) -> None:
        prev = float(self._reward_ema.get(action, 0.0))
        self._reward_ema[action] = (1.0 - float(alpha)) * prev + float(alpha) * float(reward)
        if bool(self.cfg.enable_model_memory):
            bucket = self._reward_ema_by_model.setdefault(self._model_tag, {})
            p = float(bucket.get(action, self._reward_ema[action]))
            bucket[action] = (1.0 - float(alpha)) * p + float(alpha) * float(reward)
        ctx_tag = self._normalize_context_tag(context_tag)
        ctx_bucket = self._context_bank.setdefault(ctx_tag, {})
        ctx_prev = float(ctx_bucket.get(action, self._reward_ema[action]))
        ctx_bucket[action] = (1.0 - float(alpha)) * ctx_prev + float(alpha) * float(reward)
        epi_bucket = self._episode_bank.setdefault(ctx_tag, {})
        dq = epi_bucket.setdefault(action, deque(maxlen=max(1, int(self.cfg.episode_memory_size))))
        dq.append(float(reward))

    def observe(
        self,
        *,
        info_mode: int,
        cot_mode: int,
        correctness: float,
        token_count: int,
        context_tag: Optional[str] = None,
    ) -> None:
        reward = float(correctness) - float(self.cfg.token_lambda) * (float(token_count) / 256.0)
        if bool(self.cfg.online_memory_update_enabled):
            self._update_action_reward((int(info_mode), int(cot_mode)), reward=reward, alpha=0.1, context_tag=context_tag)

    def end_episode(self, *, correctness: float, token_count: int, context_tag: Optional[str] = None) -> float:
        reward = float(correctness) - float(self.cfg.token_lambda) * (float(token_count) / 256.0)
        if bool(self.cfg.online_memory_update_enabled) and self._episode_traj:
            for pos, action in enumerate(self._episode_traj, start=1):
                decay = 0.85 ** max(0, len(self._episode_traj) - pos)
                self._update_action_reward(action, reward=reward * decay, alpha=0.08, context_tag=context_tag)
        self._episode_traj = []
        return reward

    def update_calibration(
        self,
        *,
        entropy: float,
        margin: float,
        prompt_len: int,
        generated_tokens: int = 0,
    ) -> None:
        self._entropy_stats.update(float(entropy), alpha=float(self.cfg.ema_alpha))
        self._margin_stats.update(float(margin), alpha=float(self.cfg.ema_alpha))
        self._plen_stats.update(float(prompt_len), alpha=float(self.cfg.ema_alpha))
        self._gen_tok_stats.update(float(generated_tokens), alpha=float(self.cfg.ema_alpha))

    def seed_action_memory(self, reward_by_action: Dict[Action, float]) -> None:
        for action, reward in reward_by_action.items():
            a = (int(action[0]), int(action[1]))
            if a not in self._action_space():
                continue
            r = float(reward)
            if not math.isfinite(r):
                continue
            self._reward_ema[a] = r
            if bool(self.cfg.enable_model_memory):
                self._reward_ema_by_model.setdefault(self._model_tag, {})[a] = r

    def distill_from_offline_rows(
        self,
        rows: List[Dict[str, object]],
        *,
        max_samples: int = 0,
        pair_margin: float = 0.12,
        seed: int = 20260223,
    ) -> Dict[str, float]:
        del pair_margin, seed
        if int(max_samples) > 0:
            rows = rows[: int(max_samples)]
        used = 0
        if bool(self.cfg.task_encoder_enabled):
            used = self._slot_memory.initialize_from_rows(task_encoder=self._task_encoder, rows=rows)
        if bool(self.cfg.online_memory_update_enabled):
            for row in rows:
                try:
                    action = (int(row.get("info_mode", 3)), int(row.get("cot_mode", 2)))
                    reward = float(row.get("is_correct", row.get("correctness", 0.0)) or 0.0) - float(
                        self.cfg.token_lambda
                    ) * (float(row.get("token_count", 0.0) or 0.0) / 256.0)
                except Exception:
                    continue
                if action not in self._action_space():
                    continue
                category = str(row.get("category", "")).strip().lower()
                context_tag = "code" if category == "code" else "non_code"
                self._update_action_reward(action, reward=reward, alpha=0.16, context_tag=context_tag)
                used += 1
        return {"rows": float(used), "pairs": 0.0}

    def load_state(self, path: str | Path) -> bool:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return False
        payload = json.loads(p.read_text(encoding="utf-8"))
        version = int(payload.get("version", 1))
        rem = payload.get("reward_ema", {})
        if isinstance(rem, dict):
            for k, v in rem.items():
                if not isinstance(k, str) or "_" not in k:
                    continue
                i_s, c_s = k.split("_", 1)
                action = (int(i_s), int(c_s))
                if action in self._action_space():
                    self._reward_ema[action] = float(v)
        remm = payload.get("reward_ema_by_model", {})
        if isinstance(remm, dict):
            for mk, mv in remm.items():
                if not isinstance(mk, str) or not isinstance(mv, dict):
                    continue
                dst: Dict[Action, float] = {}
                for k, v in mv.items():
                    if not isinstance(k, str) or "_" not in k:
                        continue
                    i_s, c_s = k.split("_", 1)
                    action = (int(i_s), int(c_s))
                    if action in self._action_space():
                        dst[action] = float(v)
                self._reward_ema_by_model[mk] = dst
        if version >= 2:
            ctx = payload.get("context_bank", {})
            if isinstance(ctx, dict):
                for tag, mv in ctx.items():
                    if not isinstance(tag, str) or not isinstance(mv, dict):
                        continue
                    dst: Dict[Action, float] = {}
                    for k, v in mv.items():
                        if not isinstance(k, str) or "_" not in k:
                            continue
                        i_s, c_s = k.split("_", 1)
                        action = (int(i_s), int(c_s))
                        if action in self._action_space():
                            dst[action] = float(v)
                    self._context_bank[self._normalize_context_tag(tag)] = dst
            epi = payload.get("episode_bank", {})
            if isinstance(epi, dict):
                max_epi = max(1, int(self.cfg.episode_memory_size))
                for tag, mv in epi.items():
                    if not isinstance(tag, str) or not isinstance(mv, dict):
                        continue
                    dst: Dict[Action, Deque[float]] = {}
                    for k, v in mv.items():
                        if not isinstance(k, str) or "_" not in k or not isinstance(v, list):
                            continue
                        i_s, c_s = k.split("_", 1)
                        action = (int(i_s), int(c_s))
                        if action not in self._action_space():
                            continue
                        dq: Deque[float] = deque(maxlen=max_epi)
                        for rv in v:
                            try:
                                dq.append(float(rv))
                            except Exception:
                                continue
                        dst[action] = dq
                    self._episode_bank[self._normalize_context_tag(tag)] = dst
        model_tag = str(payload.get("model_tag", "")).strip()
        if model_tag:
            self.set_model_tag(model_tag)
        head = payload.get("governance_head", {})
        if isinstance(head, dict):
            self._head.load(head)
            self._load_policy_components_from_head_metadata()

        te = payload.get("task_encoder")
        sm = payload.get("slot_memory")
        if isinstance(te, dict):
            self._task_encoder = TaskEncoder.from_dict(te)
            self.cfg.task_encoder_dim = int(self._task_encoder.dim)
            self.cfg.slot_schema_version = str(self._task_encoder.slot_schema_version)
        if isinstance(sm, dict):
            self._slot_memory = SlotMemory.from_dict(sm)
            self.cfg.target_slot_count = int(self._slot_memory.target_slot_count)
            self.cfg.slot_merge_cosine_threshold = float(self._slot_memory.merge_cosine_threshold)
            self.cfg.slot_diversity_weight = float(self._slot_memory.diversity_weight)
            self.cfg.slot_diversity_margin = float(self._slot_memory.diversity_margin)
            self.cfg.slot_match_temperature_init = float(self._slot_memory.match_temperature)
            self.cfg.slot_match_temperature_learnable = bool(self._slot_memory.match_temperature_learnable)
            self.cfg.slot_schema_version = str(self._slot_memory.slot_schema_version)
        return True

    def save_state(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 5,
            "memory_layout": str(self.cfg.memory_layout),
            "reward_ema": {f"{a[0]}_{a[1]}": float(v) for a, v in self._reward_ema.items()},
            "reward_ema_by_model": {
                mk: {f"{a[0]}_{a[1]}": float(v) for a, v in bucket.items()}
                for mk, bucket in self._reward_ema_by_model.items()
            },
            "context_bank": {
                mk: {f"{a[0]}_{a[1]}": float(v) for a, v in bucket.items()}
                for mk, bucket in self._context_bank.items()
            },
            "episode_bank": {
                mk: {f"{a[0]}_{a[1]}": [float(x) for x in list(vals)] for a, vals in bucket.items()}
                for mk, bucket in self._episode_bank.items()
            },
            "model_tag": self._model_tag,
            "governance_head": self._head.dump(),
            "task_encoder": self._task_encoder.to_dict(),
            "slot_memory": self._slot_memory.to_dict(),
            "cfg": {
                "memory_layout": self.cfg.memory_layout,
                "token_lambda": self.cfg.token_lambda,
                "global_memory_weight": self.cfg.global_memory_weight,
                "model_memory_weight": self.cfg.model_memory_weight,
                "episode_memory_weight": self.cfg.episode_memory_weight,
                "episode_memory_size": self.cfg.episode_memory_size,
                "head_scale": self.cfg.head_scale,
                "score_memory_weight": self.cfg.score_memory_weight,
                "use_hidden_observables": bool(self.cfg.use_hidden_observables),
                "online_memory_update_enabled": bool(self.cfg.online_memory_update_enabled),
                "slot_prior_weight": float(self.cfg.slot_prior_weight),
                "memory_type": str(self.cfg.memory_type),
                "task_encoder_enabled": bool(self.cfg.task_encoder_enabled),
                "task_encoder_dim": int(self.cfg.task_encoder_dim),
                "target_slot_count": int(self.cfg.target_slot_count),
                "slot_merge_cosine_threshold": float(self.cfg.slot_merge_cosine_threshold),
                "slot_diversity_weight": float(self.cfg.slot_diversity_weight),
                "slot_diversity_margin": float(self.cfg.slot_diversity_margin),
                "slot_match_temperature_init": float(self.cfg.slot_match_temperature_init),
                "slot_match_temperature_learnable": bool(self.cfg.slot_match_temperature_learnable),
                "slot_schema_version": str(self.cfg.slot_schema_version),
            },
        }
        p.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return p
