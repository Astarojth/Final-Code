from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .settings import DATASET_REGISTRY


DEFAULT_BOUNDARY_KINDS = (
    "none",
    "initial",
    "newline",
    "punct",
    "step_marker",
    "segment_budget",
    "eos",
    "answer_ready",
    "code_ready",
)


def _safe_std(x: Sequence[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return 1.0
    std = float(np.std(arr))
    return std if std > 1e-9 else 1.0


def _safe_mean(x: Sequence[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def infer_context_tag(row: Dict[str, Any]) -> str:
    category = str(row.get("category", "")).strip().lower()
    if category == "code":
        return "code"
    dataset = str(row.get("dataset", "")).strip().lower()
    spec = DATASET_REGISTRY.get(dataset)
    if spec is not None and str(spec.category).strip().lower() == "code":
        return "code"
    return "non_code"


def token_count_from_row(row: Dict[str, Any]) -> int:
    if "token_count" in row:
        try:
            return max(0, int(float(row["token_count"])))
        except Exception:
            pass
    usage = row.get("usage")
    if isinstance(usage, dict) and "completion_tokens" in usage:
        try:
            return max(0, int(float(usage["completion_tokens"])))
        except Exception:
            pass
    return 0


def compress_hidden_state(hidden_state: Sequence[float] | np.ndarray | None, proj_dim: int) -> tuple[float, ...]:
    if proj_dim <= 0:
        return ()
    if hidden_state is None:
        return tuple(0.0 for _ in range(int(proj_dim)))
    arr = np.asarray(hidden_state, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return tuple(0.0 for _ in range(int(proj_dim)))
    chunk = max(1, int(math.ceil(float(arr.size) / float(proj_dim))))
    values: List[float] = []
    for idx in range(int(proj_dim)):
        start = idx * chunk
        end = min(arr.size, start + chunk)
        if start >= arr.size:
            values.append(0.0)
            continue
        values.append(float(np.mean(arr[start:end])))
    return tuple(values)


def _normalize_hidden_proj(value: Any, *, proj_dim: int) -> tuple[float, ...]:
    if proj_dim <= 0:
        return ()
    if isinstance(value, list):
        data = [float(x) for x in value[:proj_dim]]
    elif isinstance(value, tuple):
        data = [float(x) for x in list(value)[:proj_dim]]
    else:
        data = []
    if len(data) < proj_dim:
        data.extend([0.0] * (proj_dim - len(data)))
    return tuple(data)


def _state_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("state_features")
    if isinstance(payload, dict):
        return payload
    return row


def _boundary_kind_from_row(row: Dict[str, Any]) -> str:
    payload = _state_payload(row)
    kind = str(payload.get("boundary_kind", row.get("boundary_kind", "none"))).strip().lower()
    return kind or "none"


def _hidden_proj_from_row(row: Dict[str, Any], *, proj_dim: int) -> tuple[float, ...]:
    payload = _state_payload(row)
    return _normalize_hidden_proj(
        payload.get("hidden_state_proj", row.get("probe_hidden_state_proj", row.get("hidden_state_proj"))),
        proj_dim=proj_dim,
    )


def _topk_mass_from_payload(payload: Dict[str, Any], row: Dict[str, Any]) -> float:
    value = payload.get("topk_mass_5", payload.get("topk_mass", row.get("topk_mass_5", row.get("topk_mass", 0.0))))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if not value:
            return 0.0
        if len(value) >= 5:
            return float(value[4])
        return float(value[-1])
    return 0.0


@dataclass(frozen=True)
class GovernanceFeatureEncoder:
    context_tags: tuple[str, ...]
    boundary_kinds: tuple[str, ...]
    hidden_proj_dim: int
    feature_schema_version: str
    use_hidden_observables: bool
    prompt_log_mean: float
    prompt_log_std: float
    token_log_mean: float
    token_log_std: float
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
    repeat_ngram_ratio_mean: float
    repeat_ngram_ratio_std: float
    hidden_mean: float
    hidden_std: float
    hidden_proj_mean: tuple[float, ...]
    hidden_proj_std: tuple[float, ...]

    def _use_boundary_v2(self) -> bool:
        version = str(self.feature_schema_version or "").strip().lower()
        return version != "boundary_v1"

    @classmethod
    def fit(
        cls,
        rows: Sequence[Dict[str, Any]],
        *,
        hidden_proj_dim: int = 0,
        boundary_kinds: Sequence[str] | None = None,
        feature_schema_version: str = "boundary_v2",
        use_hidden_observables: bool = True,
    ) -> "GovernanceFeatureEncoder":
        prompt_logs: List[float] = []
        token_logs: List[float] = []
        entropies: List[float] = []
        margins: List[float] = []
        top1_probs: List[float] = []
        top2_probs: List[float] = []
        topk_masses: List[float] = []
        eos_probs: List[float] = []
        eos_ranks: List[float] = []
        repeat_ngram_ratios: List[float] = []
        hidden_norms: List[float] = []
        tags = set()
        kinds = set(boundary_kinds or DEFAULT_BOUNDARY_KINDS)
        inferred_proj_dim = int(hidden_proj_dim)
        if not bool(use_hidden_observables):
            inferred_proj_dim = 0
        if inferred_proj_dim <= 0 and bool(use_hidden_observables):
            for row in rows:
                payload = _state_payload(row)
                raw_proj = payload.get("hidden_state_proj", row.get("probe_hidden_state_proj", row.get("hidden_state_proj")))
                if isinstance(raw_proj, (list, tuple)):
                    inferred_proj_dim = max(inferred_proj_dim, len(raw_proj))
        hidden_proj_cols: List[List[float]] = [[] for _ in range(max(0, inferred_proj_dim))]
        for row in rows:
            payload = _state_payload(row)
            prompt_logs.append(math.log1p(len(str(row.get("prompt", payload.get("prompt", ""))))))
            token_logs.append(
                math.log1p(
                    max(
                        0,
                        int(
                            payload.get(
                                "generated_tokens",
                                row.get("generated_tokens", token_count_from_row(row)),
                            )
                        ),
                    )
                )
            )
            entropies.append(float(payload.get("entropy", row.get("probe_entropy", row.get("entropy", 0.0)) or 0.0)))
            margins.append(float(payload.get("margin", row.get("probe_margin", row.get("margin", 0.0)) or 0.0)))
            top1_probs.append(float(payload.get("top1_prob", row.get("top1_prob", 0.0)) or 0.0))
            top2_probs.append(float(payload.get("top2_prob", row.get("top2_prob", 0.0)) or 0.0))
            topk_masses.append(_topk_mass_from_payload(payload, row))
            eos_probs.append(float(payload.get("eos_prob", row.get("eos_prob", 0.0)) or 0.0))
            eos_ranks.append(float(payload.get("eos_rank", row.get("eos_rank", 0.0)) or 0.0))
            repeat_ngram_ratios.append(
                float(payload.get("repeat_ngram_ratio", row.get("repeat_ngram_ratio", 0.0)) or 0.0)
            )
            if bool(use_hidden_observables):
                hidden_norms.append(
                    float(payload.get("hidden_l2_norm", row.get("probe_hidden_l2_norm", row.get("hidden_l2_norm", 0.0)) or 0.0))
                )
            tags.add(infer_context_tag(row))
            kinds.add(_boundary_kind_from_row(row))
            if bool(use_hidden_observables):
                hidden_proj = _hidden_proj_from_row(row, proj_dim=inferred_proj_dim)
                for idx, value in enumerate(hidden_proj):
                    hidden_proj_cols[idx].append(float(value))
        ordered_tags = tuple(sorted(tags or {"non_code", "code"}))
        ordered_kinds = tuple(sorted(kinds or DEFAULT_BOUNDARY_KINDS))
        return cls(
            context_tags=ordered_tags,
            boundary_kinds=ordered_kinds,
            hidden_proj_dim=inferred_proj_dim,
            feature_schema_version=str(feature_schema_version or "boundary_v2"),
            use_hidden_observables=bool(use_hidden_observables),
            prompt_log_mean=_safe_mean(prompt_logs),
            prompt_log_std=_safe_std(prompt_logs),
            token_log_mean=_safe_mean(token_logs),
            token_log_std=_safe_std(token_logs),
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
            repeat_ngram_ratio_mean=_safe_mean(repeat_ngram_ratios),
            repeat_ngram_ratio_std=_safe_std(repeat_ngram_ratios),
            hidden_mean=_safe_mean(hidden_norms),
            hidden_std=_safe_std(hidden_norms),
            hidden_proj_mean=tuple(_safe_mean(col) for col in hidden_proj_cols),
            hidden_proj_std=tuple(_safe_std(col) for col in hidden_proj_cols),
        )

    @classmethod
    def default(cls) -> "GovernanceFeatureEncoder":
        return cls(
            context_tags=("non_code", "code"),
            boundary_kinds=DEFAULT_BOUNDARY_KINDS,
            hidden_proj_dim=0,
            feature_schema_version="boundary_v2",
            use_hidden_observables=True,
            prompt_log_mean=0.0,
            prompt_log_std=1.0,
            token_log_mean=0.0,
            token_log_std=1.0,
            entropy_mean=0.0,
            entropy_std=1.0,
            margin_mean=0.0,
            margin_std=1.0,
            top1_prob_mean=0.0,
            top1_prob_std=1.0,
            top2_prob_mean=0.0,
            top2_prob_std=1.0,
            topk_mass_mean=0.0,
            topk_mass_std=1.0,
            eos_prob_mean=0.0,
            eos_prob_std=1.0,
            eos_rank_mean=0.0,
            eos_rank_std=1.0,
            repeat_ngram_ratio_mean=0.0,
            repeat_ngram_ratio_std=1.0,
            hidden_mean=0.0,
            hidden_std=1.0,
            hidden_proj_mean=(),
            hidden_proj_std=(),
        )

    @property
    def feature_dim(self) -> int:
        base_dim = 18 if self._use_boundary_v2() else 12
        if not bool(self.use_hidden_observables):
            base_dim -= 1  # hidden_l2_norm z-score
        proj_dim = int(self.hidden_proj_dim) if bool(self.use_hidden_observables) else 0
        numeric_dim = base_dim + proj_dim
        return len(self.context_tags) + len(self.boundary_kinds) + numeric_dim

    def _context_one_hot(self, context_tag: str) -> np.ndarray:
        vec = np.zeros((len(self.context_tags),), dtype=np.float64)
        tag = str(context_tag).strip().lower() or "non_code"
        if tag in self.context_tags:
            vec[self.context_tags.index(tag)] = 1.0
        return vec

    def _boundary_one_hot(self, boundary_kind: str) -> np.ndarray:
        vec = np.zeros((len(self.boundary_kinds),), dtype=np.float64)
        kind = str(boundary_kind).strip().lower() or "none"
        if kind in self.boundary_kinds:
            vec[self.boundary_kinds.index(kind)] = 1.0
        return vec

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
    ) -> np.ndarray:
        prompt_log = (math.log1p(max(0, int(prompt_len))) - self.prompt_log_mean) / self.prompt_log_std
        token_log = (math.log1p(max(0, int(generated_tokens))) - self.token_log_mean) / self.token_log_std
        entropy_z = (float(entropy) - self.entropy_mean) / self.entropy_std
        margin_z = (float(margin) - self.margin_mean) / self.margin_std
        hidden_z = (float(hidden_l2_norm) - self.hidden_mean) / self.hidden_std
        hidden_proj = _normalize_hidden_proj(hidden_state_proj, proj_dim=int(self.hidden_proj_dim))
        proj_numeric: List[float] = []
        if bool(self.use_hidden_observables):
            for idx in range(int(self.hidden_proj_dim)):
                mean = self.hidden_proj_mean[idx] if idx < len(self.hidden_proj_mean) else 0.0
                std = self.hidden_proj_std[idx] if idx < len(self.hidden_proj_std) else 1.0
                proj_numeric.append((float(hidden_proj[idx]) - float(mean)) / max(1e-9, float(std)))
        numeric: List[float] = [
            prompt_log,
            token_log,
            entropy_z,
            margin_z,
        ]
        if self._use_boundary_v2():
            top1_prob_z = (float(top1_prob) - self.top1_prob_mean) / self.top1_prob_std
            top2_prob_z = (float(top2_prob) - self.top2_prob_mean) / self.top2_prob_std
            topk_mass_z = (float(topk_mass) - self.topk_mass_mean) / self.topk_mass_std
            eos_prob_z = (float(eos_prob) - self.eos_prob_mean) / self.eos_prob_std
            eos_rank_z = (float(eos_rank) - self.eos_rank_mean) / self.eos_rank_std
            repeat_ngram_ratio_z = (float(repeat_ngram_ratio) - self.repeat_ngram_ratio_mean) / self.repeat_ngram_ratio_std
            numeric.extend(
                [
                    top1_prob_z,
                    top2_prob_z,
                    topk_mass_z,
                    eos_prob_z,
                    eos_rank_z,
                    repeat_ngram_ratio_z,
                ]
            )
        tail_numeric: List[float] = [
            max(0.0, min(1.0, float(progress_ratio))),
            max(0.0, min(1.0, float(current_info_mode) / 5.0)),
            max(0.0, min(1.0, float(current_cot_mode) / 3.0)),
            max(0.0, min(1.0, float(remaining_budget_ratio))),
            max(0.0, min(1.0, float(segment_progress_ratio))),
            1.0 if bool(is_answer_zone) else 0.0,
            1.0 if bool(is_code_mode) else 0.0,
        ]
        if bool(self.use_hidden_observables):
            tail_numeric = [hidden_z] + tail_numeric + proj_numeric
        numeric.extend(tail_numeric)
        numeric_arr = np.asarray(numeric, dtype=np.float64)
        return np.concatenate(
            [self._context_one_hot(context_tag), self._boundary_one_hot(boundary_kind), numeric_arr],
            axis=0,
        )

    def encode_trace(self, row: Dict[str, Any]) -> np.ndarray:
        payload = _state_payload(row)
        context_tag = str(payload.get("context_tag", infer_context_tag(row)))
        return self.encode_state(
            context_tag=context_tag,
            entropy=float(payload.get("entropy", row.get("probe_entropy", row.get("entropy", 0.0)) or 0.0)),
            margin=float(payload.get("margin", row.get("probe_margin", row.get("margin", 0.0)) or 0.0)),
            prompt_len=len(str(row.get("prompt", payload.get("prompt", "")))),
            generated_tokens=int(
                payload.get(
                    "generated_tokens",
                    row.get("generated_tokens", token_count_from_row(row)),
                )
                or 0
            ),
            progress_ratio=float(payload.get("progress_ratio", row.get("progress_ratio", 1.0) or 1.0)),
            hidden_l2_norm=float(
                payload.get(
                    "hidden_l2_norm",
                    row.get("probe_hidden_l2_norm", row.get("hidden_l2_norm", 0.0)) or 0.0,
                )
            ),
            top1_prob=float(payload.get("top1_prob", row.get("top1_prob", 0.0)) or 0.0),
            top2_prob=float(payload.get("top2_prob", row.get("top2_prob", 0.0)) or 0.0),
            topk_mass=_topk_mass_from_payload(payload, row),
            eos_prob=float(payload.get("eos_prob", row.get("eos_prob", 0.0)) or 0.0),
            eos_rank=float(payload.get("eos_rank", row.get("eos_rank", 0.0)) or 0.0),
            repeat_ngram_ratio=float(payload.get("repeat_ngram_ratio", row.get("repeat_ngram_ratio", 0.0)) or 0.0),
            current_info_mode=int(payload.get("current_info_mode", row.get("info_mode", 0)) or 0),
            current_cot_mode=int(payload.get("current_cot_mode", row.get("cot_mode", 0)) or 0),
            remaining_budget_ratio=float(payload.get("remaining_budget_ratio", row.get("remaining_budget_ratio", 1.0) or 1.0)),
            segment_progress_ratio=float(payload.get("segment_progress_ratio", row.get("segment_progress_ratio", 0.0) or 0.0)),
            is_answer_zone=bool(payload.get("is_answer_zone", row.get("is_answer_zone", False))),
            is_code_mode=bool(payload.get("is_code_mode", infer_context_tag(row) == "code")),
            boundary_kind=_boundary_kind_from_row(row),
            hidden_state_proj=_hidden_proj_from_row(row, proj_dim=int(self.hidden_proj_dim)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_tags": list(self.context_tags),
            "boundary_kinds": list(self.boundary_kinds),
            "hidden_proj_dim": int(self.hidden_proj_dim),
            "feature_schema_version": self.feature_schema_version,
            "use_hidden_observables": bool(self.use_hidden_observables),
            "prompt_log_mean": self.prompt_log_mean,
            "prompt_log_std": self.prompt_log_std,
            "token_log_mean": self.token_log_mean,
            "token_log_std": self.token_log_std,
            "entropy_mean": self.entropy_mean,
            "entropy_std": self.entropy_std,
            "margin_mean": self.margin_mean,
            "margin_std": self.margin_std,
            "top1_prob_mean": self.top1_prob_mean,
            "top1_prob_std": self.top1_prob_std,
            "top2_prob_mean": self.top2_prob_mean,
            "top2_prob_std": self.top2_prob_std,
            "topk_mass_mean": self.topk_mass_mean,
            "topk_mass_std": self.topk_mass_std,
            "eos_prob_mean": self.eos_prob_mean,
            "eos_prob_std": self.eos_prob_std,
            "eos_rank_mean": self.eos_rank_mean,
            "eos_rank_std": self.eos_rank_std,
            "repeat_ngram_ratio_mean": self.repeat_ngram_ratio_mean,
            "repeat_ngram_ratio_std": self.repeat_ngram_ratio_std,
            "hidden_mean": self.hidden_mean,
            "hidden_std": self.hidden_std,
            "hidden_proj_mean": list(self.hidden_proj_mean),
            "hidden_proj_std": list(self.hidden_proj_std),
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GovernanceFeatureEncoder":
        tags = payload.get("context_tags", ["non_code", "code"])
        if not isinstance(tags, list) or not tags:
            tags = ["non_code", "code"]
        kinds = payload.get("boundary_kinds", list(DEFAULT_BOUNDARY_KINDS))
        if not isinstance(kinds, list) or not kinds:
            kinds = list(DEFAULT_BOUNDARY_KINDS)
        hidden_proj_dim = int(payload.get("hidden_proj_dim", 0) or 0)
        hidden_proj_mean = payload.get("hidden_proj_mean", [])
        hidden_proj_std = payload.get("hidden_proj_std", [])
        if not isinstance(hidden_proj_mean, list):
            hidden_proj_mean = []
        if not isinstance(hidden_proj_std, list):
            hidden_proj_std = []
        if len(hidden_proj_mean) < hidden_proj_dim:
            hidden_proj_mean = list(hidden_proj_mean) + [0.0] * (hidden_proj_dim - len(hidden_proj_mean))
        if len(hidden_proj_std) < hidden_proj_dim:
            hidden_proj_std = list(hidden_proj_std) + [1.0] * (hidden_proj_dim - len(hidden_proj_std))
        return cls(
            context_tags=tuple(str(x) for x in tags),
            boundary_kinds=tuple(str(x) for x in kinds),
            hidden_proj_dim=hidden_proj_dim,
            feature_schema_version=str(payload.get("feature_schema_version", "boundary_v2")),
            use_hidden_observables=bool(payload.get("use_hidden_observables", True)),
            prompt_log_mean=float(payload.get("prompt_log_mean", 0.0)),
            prompt_log_std=max(1e-9, float(payload.get("prompt_log_std", 1.0))),
            token_log_mean=float(payload.get("token_log_mean", 0.0)),
            token_log_std=max(1e-9, float(payload.get("token_log_std", 1.0))),
            entropy_mean=float(payload.get("entropy_mean", 0.0)),
            entropy_std=max(1e-9, float(payload.get("entropy_std", 1.0))),
            margin_mean=float(payload.get("margin_mean", 0.0)),
            margin_std=max(1e-9, float(payload.get("margin_std", 1.0))),
            top1_prob_mean=float(payload.get("top1_prob_mean", 0.0)),
            top1_prob_std=max(1e-9, float(payload.get("top1_prob_std", 1.0))),
            top2_prob_mean=float(payload.get("top2_prob_mean", 0.0)),
            top2_prob_std=max(1e-9, float(payload.get("top2_prob_std", 1.0))),
            topk_mass_mean=float(payload.get("topk_mass_mean", 0.0)),
            topk_mass_std=max(1e-9, float(payload.get("topk_mass_std", 1.0))),
            eos_prob_mean=float(payload.get("eos_prob_mean", 0.0)),
            eos_prob_std=max(1e-9, float(payload.get("eos_prob_std", 1.0))),
            eos_rank_mean=float(payload.get("eos_rank_mean", 0.0)),
            eos_rank_std=max(1e-9, float(payload.get("eos_rank_std", 1.0))),
            repeat_ngram_ratio_mean=float(payload.get("repeat_ngram_ratio_mean", 0.0)),
            repeat_ngram_ratio_std=max(1e-9, float(payload.get("repeat_ngram_ratio_std", 1.0))),
            hidden_mean=float(payload.get("hidden_mean", 0.0)),
            hidden_std=max(1e-9, float(payload.get("hidden_std", 1.0))),
            hidden_proj_mean=tuple(float(x) for x in hidden_proj_mean[:hidden_proj_dim]),
            hidden_proj_std=tuple(max(1e-9, float(x)) for x in hidden_proj_std[:hidden_proj_dim]),
        )


@dataclass(frozen=True)
class DualHeadGovernanceModel:
    trunk_weight: np.ndarray
    trunk_bias: np.ndarray
    info_weight: np.ndarray
    info_bias: np.ndarray
    cot_weight: np.ndarray
    cot_bias: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.trunk_weight.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.trunk_weight.shape[1])

    def forward(self, feat: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.asarray(feat, dtype=np.float64).reshape(-1)
        h = np.tanh((x @ self.trunk_weight) + self.trunk_bias)
        info_logits = (h @ self.info_weight) + self.info_bias
        cot_logits = (h @ self.cot_weight) + self.cot_bias
        return {
            "hidden": h,
            "info_logits": info_logits,
            "cot_logits": cot_logits,
            "info_log_probs": _log_softmax(info_logits),
            "cot_log_probs": _log_softmax(cot_logits),
        }

    def action_score(self, feat: np.ndarray, *, info_mode: int, cot_mode: int) -> float:
        out = self.forward(feat)
        i = max(0, min(4, int(info_mode) - 1))
        c = max(0, min(3, int(cot_mode)))
        return float(out["info_log_probs"][i] + out["cot_log_probs"][c])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trunk_weight": self.trunk_weight.tolist(),
            "trunk_bias": self.trunk_bias.tolist(),
            "info_weight": self.info_weight.tolist(),
            "info_bias": self.info_bias.tolist(),
            "cot_weight": self.cot_weight.tolist(),
            "cot_bias": self.cot_bias.tolist(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DualHeadGovernanceModel":
        return cls(
            trunk_weight=np.asarray(payload["trunk_weight"], dtype=np.float64),
            trunk_bias=np.asarray(payload["trunk_bias"], dtype=np.float64),
            info_weight=np.asarray(payload["info_weight"], dtype=np.float64),
            info_bias=np.asarray(payload["info_bias"], dtype=np.float64),
            cot_weight=np.asarray(payload["cot_weight"], dtype=np.float64),
            cot_bias=np.asarray(payload["cot_bias"], dtype=np.float64),
        )


def _log_softmax(x: np.ndarray) -> np.ndarray:
    vec = np.asarray(x, dtype=np.float64).reshape(-1)
    shift = vec - np.max(vec)
    exp = np.exp(shift)
    return shift - math.log(float(np.sum(exp)))


def save_governance_bundle(
    path: str | Path,
    *,
    encoder: GovernanceFeatureEncoder,
    model: DualHeadGovernanceModel,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "encoder": encoder.to_dict(),
        "model": model.to_dict(),
        "metadata": metadata or {},
    }
    dst.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return dst


def write_governance_bundle(
    path: str | Path,
    *,
    encoder: GovernanceFeatureEncoder,
    model: DualHeadGovernanceModel,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    return save_governance_bundle(path, encoder=encoder, model=model, metadata=metadata)


def load_governance_bundle(path: str | Path) -> Dict[str, Any]:
    src = Path(path).expanduser().resolve()
    payload = json.loads(src.read_text(encoding="utf-8"))
    return {
        "version": int(payload.get("version", 1)),
        "encoder": GovernanceFeatureEncoder.from_dict(payload["encoder"]),
        "model": DualHeadGovernanceModel.from_dict(payload["model"]),
        "metadata": payload.get("metadata", {}),
        "path": str(src),
    }


def random_governance_model(*, input_dim: int, hidden_dim: int = 16, seed: int = 42) -> DualHeadGovernanceModel:
    rng = np.random.default_rng(int(seed))
    trunk_weight = rng.normal(0.0, 0.1, size=(int(input_dim), int(hidden_dim)))
    trunk_bias = rng.normal(0.0, 0.05, size=(int(hidden_dim),))
    info_weight = rng.normal(0.0, 0.1, size=(int(hidden_dim), 5))
    info_bias = rng.normal(0.0, 0.05, size=(5,))
    cot_weight = rng.normal(0.0, 0.1, size=(int(hidden_dim), 4))
    cot_bias = rng.normal(0.0, 0.05, size=(4,))
    return DualHeadGovernanceModel(
        trunk_weight=trunk_weight.astype(np.float64),
        trunk_bias=trunk_bias.astype(np.float64),
        info_weight=info_weight.astype(np.float64),
        info_bias=info_bias.astype(np.float64),
        cot_weight=cot_weight.astype(np.float64),
        cot_bias=cot_bias.astype(np.float64),
    )
