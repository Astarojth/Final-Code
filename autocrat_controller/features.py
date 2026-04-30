from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return 1.0
    std = float(arr.std())
    return std if std > 1e-6 else 1.0


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % max(1, int(dim))


@dataclass(frozen=True)
class HashTextVectorizer:
    dim: int = 1024
    use_char_ngrams: bool = True

    def _tokens(self, text: str) -> Iterable[str]:
        lowered = text.lower()
        for tok in re.findall(r"\w+|[^\w\s]", lowered, flags=re.UNICODE):
            yield f"w:{tok}"
        if self.use_char_ngrams:
            compact = re.sub(r"\s+", " ", lowered).strip()
            for n in (3, 4, 5):
                if len(compact) < n:
                    continue
                for i in range(len(compact) - n + 1):
                    yield f"c{n}:{compact[i:i+n]}"

    def transform_one(self, text: str) -> np.ndarray:
        vec = np.zeros(int(self.dim), dtype=np.float32)
        if not text:
            return vec
        for token in self._tokens(str(text)):
            idx = _stable_bucket(token, self.dim)
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self.transform_one(text) for text in texts]
        if not rows:
            return np.zeros((0, int(self.dim)), dtype=np.float32)
        return np.stack(rows, axis=0)


@dataclass(frozen=True)
class BoundaryFeatureSpec:
    boundary_kinds: Tuple[str, ...]
    prompt_log_mean: float
    prompt_log_std: float
    generated_log_mean: float
    generated_log_std: float
    entropy_mean: float
    entropy_std: float
    margin_mean: float
    margin_std: float
    top1_prob_mean: float
    top1_prob_std: float
    top2_prob_mean: float
    top2_prob_std: float
    topk_mass_mean: float
    topk_mass_std: float
    eos_prob_mean: float
    eos_prob_std: float
    eos_rank_mean: float
    eos_rank_std: float
    repeat_ngram_mean: float
    repeat_ngram_std: float
    progress_mean: float
    progress_std: float
    remaining_budget_mean: float
    remaining_budget_std: float
    segment_progress_mean: float
    segment_progress_std: float

    @property
    def dim(self) -> int:
        return 16 + len(self.boundary_kinds)

    @classmethod
    def fit(cls, rows: Sequence[Dict[str, Any]]) -> "BoundaryFeatureSpec":
        kinds = set()
        prompt_logs: List[float] = []
        generated_logs: List[float] = []
        entropies: List[float] = []
        margins: List[float] = []
        top1_probs: List[float] = []
        top2_probs: List[float] = []
        topk_masses: List[float] = []
        eos_probs: List[float] = []
        eos_ranks: List[float] = []
        repeat_ratios: List[float] = []
        progresses: List[float] = []
        remaining_budgets: List[float] = []
        segment_progresses: List[float] = []
        for row in rows:
            state = row.get("state_features", row)
            if not isinstance(state, dict):
                state = {}
            kinds.add(str(state.get("boundary_kind", row.get("boundary_kind", "none")) or "none"))
            prompt_logs.append(math.log1p(len(str(row.get("prompt", state.get("prompt", ""))))))
            generated_logs.append(
                math.log1p(float(state.get("generated_tokens", row.get("token_count", 0) or 0) or 0.0))
            )
            entropies.append(float(state.get("entropy", 0.0) or 0.0))
            margins.append(float(state.get("margin", 0.0) or 0.0))
            top1_probs.append(float(state.get("top1_prob", 0.0) or 0.0))
            top2_probs.append(float(state.get("top2_prob", 0.0) or 0.0))
            topk_masses.append(float(state.get("topk_mass", 0.0) or 0.0))
            eos_probs.append(float(state.get("eos_prob", 0.0) or 0.0))
            eos_ranks.append(float(state.get("eos_rank", 0.0) or 0.0))
            repeat_ratios.append(float(state.get("repeat_ngram_ratio", 0.0) or 0.0))
            progresses.append(float(state.get("progress_ratio", 0.0) or 0.0))
            remaining_budgets.append(float(state.get("remaining_budget_ratio", 0.0) or 0.0))
            segment_progresses.append(float(state.get("segment_progress_ratio", 0.0) or 0.0))
        ordered_kinds = tuple(sorted(kinds or {"none"}))
        return cls(
            boundary_kinds=ordered_kinds,
            prompt_log_mean=_safe_mean(prompt_logs),
            prompt_log_std=_safe_std(prompt_logs),
            generated_log_mean=_safe_mean(generated_logs),
            generated_log_std=_safe_std(generated_logs),
            entropy_mean=_safe_mean(entropies),
            entropy_std=_safe_std(entropies),
            margin_mean=_safe_mean(margins),
            margin_std=_safe_std(margins),
            top1_prob_mean=_safe_mean(top1_probs),
            top1_prob_std=_safe_std(top1_probs),
            top2_prob_mean=_safe_mean(top2_probs),
            top2_prob_std=_safe_std(top2_probs),
            topk_mass_mean=_safe_mean(topk_masses),
            topk_mass_std=_safe_std(topk_masses),
            eos_prob_mean=_safe_mean(eos_probs),
            eos_prob_std=_safe_std(eos_probs),
            eos_rank_mean=_safe_mean(eos_ranks),
            eos_rank_std=_safe_std(eos_ranks),
            repeat_ngram_mean=_safe_mean(repeat_ratios),
            repeat_ngram_std=_safe_std(repeat_ratios),
            progress_mean=_safe_mean(progresses),
            progress_std=_safe_std(progresses),
            remaining_budget_mean=_safe_mean(remaining_budgets),
            remaining_budget_std=_safe_std(remaining_budgets),
            segment_progress_mean=_safe_mean(segment_progresses),
            segment_progress_std=_safe_std(segment_progresses),
        )

    def transform_one(self, row: Dict[str, Any]) -> np.ndarray:
        state = row.get("state_features", row)
        if not isinstance(state, dict):
            state = {}
        out: List[float] = []
        out.append(
            (math.log1p(len(str(row.get("prompt", state.get("prompt", ""))))) - self.prompt_log_mean)
            / self.prompt_log_std
        )
        out.append(
            (math.log1p(float(state.get("generated_tokens", row.get("token_count", 0) or 0) or 0.0)) - self.generated_log_mean)
            / self.generated_log_std
        )
        out.append((float(state.get("entropy", 0.0) or 0.0) - self.entropy_mean) / self.entropy_std)
        out.append((float(state.get("margin", 0.0) or 0.0) - self.margin_mean) / self.margin_std)
        out.append((float(state.get("top1_prob", 0.0) or 0.0) - self.top1_prob_mean) / self.top1_prob_std)
        out.append((float(state.get("top2_prob", 0.0) or 0.0) - self.top2_prob_mean) / self.top2_prob_std)
        out.append((float(state.get("topk_mass", 0.0) or 0.0) - self.topk_mass_mean) / self.topk_mass_std)
        out.append((float(state.get("eos_prob", 0.0) or 0.0) - self.eos_prob_mean) / self.eos_prob_std)
        out.append((float(state.get("eos_rank", 0.0) or 0.0) - self.eos_rank_mean) / self.eos_rank_std)
        out.append(
            (float(state.get("repeat_ngram_ratio", 0.0) or 0.0) - self.repeat_ngram_mean)
            / self.repeat_ngram_std
        )
        out.append((float(state.get("progress_ratio", 0.0) or 0.0) - self.progress_mean) / self.progress_std)
        out.append(
            (float(state.get("remaining_budget_ratio", 0.0) or 0.0) - self.remaining_budget_mean)
            / self.remaining_budget_std
        )
        out.append(
            (float(state.get("segment_progress_ratio", 0.0) or 0.0) - self.segment_progress_mean)
            / self.segment_progress_std
        )
        out.append(_bool(state.get("is_answer_zone", False)))
        out.append(_bool(state.get("is_in_think_zone", False)))
        out.append(_bool(state.get("is_code_mode", False)))
        kind = str(state.get("boundary_kind", row.get("boundary_kind", "none")) or "none")
        out.extend([1.0 if kind == candidate else 0.0 for candidate in self.boundary_kinds])
        return np.asarray(out, dtype=np.float32)

    def transform(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.transform_one(row) for row in rows], axis=0)
