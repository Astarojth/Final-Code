from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        denom = np.linalg.norm(x)
        return x / max(1e-8, float(denom))
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(1e-8, denom)
    return x / denom


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-6, float(temperature))
    z = x / t
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.maximum(1e-8, exp_z.sum())


def kmeans(x: np.ndarray, num_clusters: int, seed: int, steps: int = 25) -> np.ndarray:
    if x.shape[0] <= num_clusters:
        return x.copy()
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(x.shape[0], size=int(num_clusters), replace=False)
    centers = x[indices].copy()
    for _ in range(int(steps)):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for k in range(int(num_clusters)):
            mask = assign == k
            if np.any(mask):
                new_centers[k] = x[mask].mean(axis=0)
        shift = float(np.abs(new_centers - centers).mean())
        centers = new_centers
        if shift < 1e-5:
            break
    return centers


@dataclass(frozen=True)
class SlotMemory:
    centers: np.ndarray
    priors: np.ndarray
    temperature: float

    @classmethod
    def fit(
        cls,
        task_vectors: np.ndarray,
        preference_distributions: np.ndarray,
        *,
        num_slots: int,
        seed: int = 42,
        temperature: float = 0.35,
        example_weights: np.ndarray | None = None,
    ) -> "SlotMemory":
        if task_vectors.ndim != 2:
            raise ValueError("task_vectors must be rank-2")
        if preference_distributions.ndim != 2:
            raise ValueError("preference_distributions must be rank-2")
        if task_vectors.shape[0] != preference_distributions.shape[0]:
            raise ValueError("task_vectors and preference_distributions must have same row count")

        x = _l2_normalize(task_vectors.astype(np.float32))
        centers = kmeans(x, num_clusters=min(int(num_slots), x.shape[0]), seed=seed)
        centers = _l2_normalize(centers.astype(np.float32))

        sims = x @ centers.T
        assign = np.argmax(sims, axis=1)
        priors = np.zeros((centers.shape[0], preference_distributions.shape[1]), dtype=np.float32)
        weights = np.ones((x.shape[0],), dtype=np.float32) if example_weights is None else example_weights.astype(np.float32)
        for slot_idx in range(centers.shape[0]):
            mask = assign == slot_idx
            if not np.any(mask):
                priors[slot_idx] = preference_distributions.mean(axis=0)
                continue
            slot_w = weights[mask][:, None]
            weighted = preference_distributions[mask] * slot_w
            priors[slot_idx] = weighted.sum(axis=0) / np.maximum(1e-8, slot_w.sum())
        priors = np.asarray([_softmax(p, temperature=1.0) for p in priors], dtype=np.float32)
        return cls(centers=centers, priors=priors, temperature=float(temperature))

    def query(self, task_vectors: np.ndarray) -> np.ndarray:
        x = _l2_normalize(task_vectors.astype(np.float32))
        out = []
        for row in x:
            sims = row @ self.centers.T
            slot_weights = _softmax(sims, temperature=self.temperature)
            out.append(slot_weights @ self.priors)
        if not out:
            return np.zeros((0, self.priors.shape[1]), dtype=np.float32)
        return np.stack(out, axis=0).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "priors": self.priors.tolist(),
            "temperature": float(self.temperature),
        }
