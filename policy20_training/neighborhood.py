from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _state_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    state = row.get("state_features", row)
    return state if isinstance(state, dict) else row


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    return float(arr.mean()) if arr.size else 0.0


def _safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return 1.0
    std = float(arr.std())
    return std if std > 1e-6 else 1.0


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


@dataclass(frozen=True)
class NeighborhoodFeature:
    name: str
    kind: str
    weight: float
    enabled: bool = True


@dataclass(frozen=True)
class BoundaryNeighborhoodConfig:
    features: Tuple[NeighborhoodFeature, ...]
    top_k: int = 8
    kernel_temperature: float = 0.75
    coarse_bucket_strategy: str = "none"
    progress_bucket_count: int = 16
    progress_bucket_radius: int = 1
    boundary_index_bucket_size: int = 8
    boundary_index_bucket_radius: int = 1
    require_same_answer_zone: bool = True
    require_same_boundary_kind: bool = False
    min_kernel_weight: float = 1e-4

    def enabled_features(self) -> Tuple[NeighborhoodFeature, ...]:
        return tuple(feature for feature in self.features if feature.enabled and feature.weight > 0.0)


@dataclass(frozen=True)
class BoundaryNeighborhoodSpec:
    config: BoundaryNeighborhoodConfig
    numeric_stats: Mapping[str, Tuple[float, float]]

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any]],
        config: BoundaryNeighborhoodConfig,
    ) -> "BoundaryNeighborhoodSpec":
        numeric_values: Dict[str, List[float]] = {}
        for feature in config.enabled_features():
            if feature.kind == "numeric":
                numeric_values[feature.name] = []
        for row in rows:
            state = _state_payload(row)
            for feature in config.enabled_features():
                if feature.kind != "numeric":
                    continue
                numeric_values[feature.name].append(_numeric_feature_value(feature.name, state, row))
        stats = {
            name: (_safe_mean(values), _safe_std(values))
            for name, values in numeric_values.items()
        }
        return cls(config=config, numeric_stats=stats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "features": [asdict(feature) for feature in self.config.features],
                "top_k": int(self.config.top_k),
                "kernel_temperature": float(self.config.kernel_temperature),
                "coarse_bucket_strategy": str(self.config.coarse_bucket_strategy),
                "progress_bucket_count": int(self.config.progress_bucket_count),
                "progress_bucket_radius": int(self.config.progress_bucket_radius),
                "boundary_index_bucket_size": int(self.config.boundary_index_bucket_size),
                "boundary_index_bucket_radius": int(self.config.boundary_index_bucket_radius),
                "require_same_answer_zone": bool(self.config.require_same_answer_zone),
                "require_same_boundary_kind": bool(self.config.require_same_boundary_kind),
                "min_kernel_weight": float(self.config.min_kernel_weight),
            },
            "numeric_stats": {
                name: {"mean": float(mean), "std": float(std)}
                for name, (mean, std) in self.numeric_stats.items()
            },
        }

    def candidate_weight(
        self,
        anchor_row: Mapping[str, Any],
        candidate_row: Mapping[str, Any],
    ) -> float:
        anchor_state = _state_payload(anchor_row)
        candidate_state = _state_payload(candidate_row)

        if self.config.require_same_answer_zone:
            if bool(anchor_state.get("is_answer_zone", anchor_row.get("is_answer_zone", False))) != bool(
                candidate_state.get("is_answer_zone", candidate_row.get("is_answer_zone", False))
            ):
                return 0.0
        if self.config.require_same_boundary_kind:
            if str(anchor_state.get("boundary_kind", anchor_row.get("boundary_kind", "none")) or "none") != str(
                candidate_state.get("boundary_kind", candidate_row.get("boundary_kind", "none")) or "none"
            ):
                return 0.0

        distance = 0.0
        for feature in self.config.enabled_features():
            distance += self._feature_distance(feature, anchor_row, candidate_row, anchor_state, candidate_state)

        temperature = max(1e-6, float(self.config.kernel_temperature))
        weight = math.exp(-distance / temperature)
        return weight if weight >= float(self.config.min_kernel_weight) else 0.0

    def coarse_bucket_id(self, row: Mapping[str, Any]) -> Tuple[str, int]:
        state = _state_payload(row)
        strategy = str(self.config.coarse_bucket_strategy or "none")
        if strategy == "none":
            return ("none", 0)
        if strategy == "boundary_index":
            bucket_size = max(1, int(self.config.boundary_index_bucket_size))
            boundary_index = int(row.get("boundary_index", 0) or 0)
            return ("boundary_index", boundary_index // bucket_size)
        progress_bucket_count = max(1, int(self.config.progress_bucket_count))
        progress = float(state.get("progress_ratio", row.get("progress_ratio", 0.0)) or 0.0)
        progress = min(1.0, max(0.0, progress))
        bucket = min(progress_bucket_count - 1, int(progress * progress_bucket_count))
        return ("progress_ratio", bucket)

    def neighboring_bucket_ids(self, row: Mapping[str, Any]) -> Tuple[Tuple[str, int], ...]:
        bucket_type, bucket_value = self.coarse_bucket_id(row)
        if bucket_type == "boundary_index":
            radius = max(0, int(self.config.boundary_index_bucket_radius))
        else:
            radius = max(0, int(self.config.progress_bucket_radius))
        return tuple((bucket_type, bucket_value + offset) for offset in range(-radius, radius + 1))

    def _feature_distance(
        self,
        feature: NeighborhoodFeature,
        anchor_row: Mapping[str, Any],
        candidate_row: Mapping[str, Any],
        anchor_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> float:
        if feature.kind == "numeric":
            anchor_v = _numeric_feature_value(feature.name, anchor_state, anchor_row)
            candidate_v = _numeric_feature_value(feature.name, candidate_state, candidate_row)
            mean, std = self.numeric_stats.get(feature.name, (0.0, 1.0))
            return float(feature.weight) * abs((anchor_v - mean) / std - (candidate_v - mean) / std)
        if feature.kind == "binary":
            anchor_v = _bool(anchor_state.get(feature.name, anchor_row.get(feature.name, False)))
            candidate_v = _bool(candidate_state.get(feature.name, candidate_row.get(feature.name, False)))
            return float(feature.weight) * abs(anchor_v - candidate_v)
        if feature.kind == "categorical":
            anchor_v = str(anchor_state.get(feature.name, anchor_row.get(feature.name, "none")) or "none")
            candidate_v = str(candidate_state.get(feature.name, candidate_row.get(feature.name, "none")) or "none")
            return 0.0 if anchor_v == candidate_v else float(feature.weight)
        raise ValueError(f"Unsupported neighborhood feature kind: {feature.kind}")


def _numeric_feature_value(name: str, state: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    raw = state.get(name, row.get(name, 0.0))
    if name == "generated_tokens":
        return math.log1p(float(raw or 0.0))
    return float(raw or 0.0)


DEFAULT_NEIGHBORHOOD_CONFIG = BoundaryNeighborhoodConfig(
    features=(
        NeighborhoodFeature("progress_ratio", "numeric", 1.20),
        NeighborhoodFeature("remaining_budget_ratio", "numeric", 0.90),
        NeighborhoodFeature("segment_progress_ratio", "numeric", 0.75),
        NeighborhoodFeature("generated_tokens", "numeric", 0.70),
        NeighborhoodFeature("entropy", "numeric", 0.80),
        NeighborhoodFeature("margin", "numeric", 0.60),
        NeighborhoodFeature("top1_prob", "numeric", 0.60),
        NeighborhoodFeature("repeat_ngram_ratio", "numeric", 0.55),
        NeighborhoodFeature("boundary_kind", "categorical", 1.10),
        NeighborhoodFeature("is_answer_zone", "binary", 2.00),
        NeighborhoodFeature("is_in_think_zone", "binary", 1.25),
    ),
    top_k=8,
    kernel_temperature=0.75,
    coarse_bucket_strategy="none",
    require_same_answer_zone=True,
    require_same_boundary_kind=False,
    min_kernel_weight=1e-4,
)
