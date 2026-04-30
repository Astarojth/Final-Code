from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .governance_model import (
    DualHeadGovernanceModel,
    GovernanceFeatureEncoder,
    save_governance_bundle,
)

FeatureEncoder = GovernanceFeatureEncoder
REQUIRED_GOV_CONTEXT_KEYS = (
    "prompt",
    "context_tag",
    "entropy",
    "margin",
    "prompt_len",
    "generated_tokens",
    "progress_ratio",
    "current_info_mode",
    "current_cot_mode",
    "remaining_budget_ratio",
    "segment_progress_ratio",
    "is_answer_zone",
    "is_code_mode",
    "boundary_kind",
)


def _problem_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row["dataset"]), str(row["problem_id"])


@dataclass(frozen=True)
class PreferenceDataset:
    x_pos: np.ndarray
    y_info_pos: np.ndarray
    y_cot_pos: np.ndarray
    x_neg: np.ndarray
    y_info_neg: np.ndarray
    y_cot_neg: np.ndarray
    sample_weights: np.ndarray


@dataclass(frozen=True)
class DualHeadTrainingConfig:
    hidden_dim: int = 16
    steps: int = 400
    lr: float = 0.03
    weight_decay: float = 1e-4
    preference_weight: float = 0.5
    slot_diversity_weight: float = 0.0
    slot_diversity_loss: float = 0.0
    seed: int = 42


def _validate_governance_context_rows(rows: List[Dict[str, Any]]) -> None:
    for idx, row in enumerate(rows, start=1):
        payload = row.get("state_features")
        if not isinstance(payload, dict):
            raise ValueError(f"Boundary row missing state_features dict at index={idx}")
        missing = [key for key in REQUIRED_GOV_CONTEXT_KEYS if key not in payload]
        if missing:
            raise ValueError(f"Boundary row missing governance context keys={missing} at index={idx}")


def build_bc_dataset(
    rows: List[Dict[str, Any]],
    encoder: FeatureEncoder,
    *,
    supervision_mode: str = "trajectory",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if supervision_mode == "boundary":
        _validate_governance_context_rows(rows)
        xs = [encoder.encode_trace(row) for row in rows]
        ys_info = [int(row["chosen_info_mode"]) - 1 for row in rows]
        ys_cot = [int(row["chosen_cot_mode"]) for row in rows]
        return np.stack(xs), np.asarray(ys_info, dtype=np.int64), np.asarray(ys_cot, dtype=np.int64)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_problem_key(row), []).append(row)

    xs = []
    ys_info = []
    ys_cot = []
    for group_rows in grouped.values():
        best = max(group_rows, key=lambda r: float(r["score"]))
        xs.append(encoder.encode_trace(best))
        ys_info.append(int(best["info_mode"]) - 1)
        ys_cot.append(int(best["cot_mode"]))
    return np.stack(xs), np.asarray(ys_info, dtype=np.int64), np.asarray(ys_cot, dtype=np.int64)


def build_preference_dataset(
    rows: List[Dict[str, Any]],
    preference_pairs: List[Dict[str, Any]],
    encoder: FeatureEncoder,
    *,
    supervision_mode: str = "trajectory",
) -> PreferenceDataset:
    if supervision_mode == "boundary":
        _validate_governance_context_rows(rows)
    index: Dict[str, Dict[str, Any]] = {}
    if supervision_mode == "boundary":
        for row in rows:
            boundary_id = f"{row['trace_id']}#{int(row['boundary_index'])}"
            if boundary_id in index:
                raise ValueError(f"Duplicate boundary sample id: {boundary_id}")
            index[boundary_id] = row
    else:
        for row in rows:
            trace_id = str(row["trace_id"])
            if trace_id in index:
                raise ValueError(f"Duplicate trace_id: {trace_id}")
            index[trace_id] = row

    x_pos: List[np.ndarray] = []
    y_info_pos: List[int] = []
    y_cot_pos: List[int] = []
    x_neg: List[np.ndarray] = []
    y_info_neg: List[int] = []
    y_cot_neg: List[int] = []
    wts: List[float] = []

    for pair in preference_pairs:
        if supervision_mode == "boundary":
            win_id = str(pair["winner_boundary_id"])
            lose_id = str(pair["loser_boundary_id"])
            y_info_pos.append(int(pair["winner_mode"]["info_mode"]) - 1)
            y_cot_pos.append(int(pair["winner_mode"]["cot_mode"]))
            y_info_neg.append(int(pair["loser_mode"]["info_mode"]) - 1)
            y_cot_neg.append(int(pair["loser_mode"]["cot_mode"]))
        else:
            win_id = str(pair["winner_trace_id"])
            lose_id = str(pair["loser_trace_id"])
            y_info_pos.append(int(pair["winner_mode"]["info_mode"]) - 1)
            y_cot_pos.append(int(pair["winner_mode"]["cot_mode"]))
            y_info_neg.append(int(pair["loser_mode"]["info_mode"]) - 1)
            y_cot_neg.append(int(pair["loser_mode"]["cot_mode"]))
        if win_id not in index or lose_id not in index:
            raise KeyError(f"Preference pair references unknown sample id: {win_id}, {lose_id}")
        x_pos.append(encoder.encode_trace(index[win_id]))
        x_neg.append(encoder.encode_trace(index[lose_id]))
        wt = float(pair.get("pair_weight", 1.0))
        if wt <= 0.0 or not np.isfinite(wt):
            wt = 1.0
        wts.append(wt)

    if not x_pos:
        dim = encoder.feature_dim
        return PreferenceDataset(
            x_pos=np.zeros((0, dim), dtype=np.float64),
            y_info_pos=np.zeros((0,), dtype=np.int64),
            y_cot_pos=np.zeros((0,), dtype=np.int64),
            x_neg=np.zeros((0, dim), dtype=np.float64),
            y_info_neg=np.zeros((0,), dtype=np.int64),
            y_cot_neg=np.zeros((0,), dtype=np.int64),
            sample_weights=np.zeros((0,), dtype=np.float64),
        )

    return PreferenceDataset(
        x_pos=np.stack(x_pos),
        y_info_pos=np.asarray(y_info_pos, dtype=np.int64),
        y_cot_pos=np.asarray(y_cot_pos, dtype=np.int64),
        x_neg=np.stack(x_neg),
        y_info_neg=np.asarray(y_info_neg, dtype=np.int64),
        y_cot_neg=np.asarray(y_cot_neg, dtype=np.int64),
        sample_weights=np.asarray(wts, dtype=np.float64),
    )


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Dual-head governance training requires torch.") from exc
    return torch


def train_dual_head_model(
    x_bc: np.ndarray,
    y_info: np.ndarray,
    y_cot: np.ndarray,
    preference_data: PreferenceDataset | np.ndarray,
    *args: Any,
    hidden_dim: int = 16,
    steps: int = 400,
    lr: float = 0.03,
    l2: float = 1e-4,
    preference_weight: float = 0.5,
    slot_diversity_weight: float = 0.0,
    slot_diversity_loss: float = 0.0,
    seed: int = 42,
    cfg: DualHeadTrainingConfig | None = None,
) -> Dict[str, Any]:
    if cfg is not None:
        hidden_dim = int(cfg.hidden_dim)
        steps = int(cfg.steps)
        lr = float(cfg.lr)
        l2 = float(cfg.weight_decay)
        preference_weight = float(cfg.preference_weight)
        slot_diversity_weight = float(cfg.slot_diversity_weight)
        slot_diversity_loss = float(cfg.slot_diversity_loss)
        seed = int(cfg.seed)

    if isinstance(preference_data, np.ndarray):
        if len(args) != 6:
            raise ValueError("Expected x_pos/y_pos/x_neg/y_neg/sample_weights arguments for tuple-style training call.")
        preference_data = PreferenceDataset(
            x_pos=np.asarray(preference_data, dtype=np.float64),
            y_info_pos=np.asarray(args[0], dtype=np.int64),
            y_cot_pos=np.asarray(args[1], dtype=np.int64),
            x_neg=np.asarray(args[2], dtype=np.float64),
            y_info_neg=np.asarray(args[3], dtype=np.int64),
            y_cot_neg=np.asarray(args[4], dtype=np.int64),
            sample_weights=np.asarray(args[5], dtype=np.float64),
        )

    torch = _load_torch()
    torch.manual_seed(int(seed))

    class TinyDualHead(torch.nn.Module):
        def __init__(self, input_dim: int, trunk_hidden_dim: int) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(input_dim, trunk_hidden_dim),
                torch.nn.Tanh(),
            )
            self.info_head = torch.nn.Linear(trunk_hidden_dim, 5)
            self.cot_head = torch.nn.Linear(trunk_hidden_dim, 4)

        def forward(self, x: Any) -> Tuple[Any, Any]:
            h = self.trunk(x)
            return self.info_head(h), self.cot_head(h)

    device = torch.device("cpu")
    model = TinyDualHead(int(x_bc.shape[1]), int(hidden_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(l2))
    x_bc_t = torch.tensor(x_bc, dtype=torch.float32, device=device)
    y_info_t = torch.tensor(y_info, dtype=torch.long, device=device)
    y_cot_t = torch.tensor(y_cot, dtype=torch.long, device=device)

    x_pos_t = torch.tensor(preference_data.x_pos, dtype=torch.float32, device=device)
    x_neg_t = torch.tensor(preference_data.x_neg, dtype=torch.float32, device=device)
    y_info_pos_t = torch.tensor(preference_data.y_info_pos, dtype=torch.long, device=device)
    y_cot_pos_t = torch.tensor(preference_data.y_cot_pos, dtype=torch.long, device=device)
    y_info_neg_t = torch.tensor(preference_data.y_info_neg, dtype=torch.long, device=device)
    y_cot_neg_t = torch.tensor(preference_data.y_cot_neg, dtype=torch.long, device=device)
    pair_w_t = torch.tensor(preference_data.sample_weights, dtype=torch.float32, device=device)

    for _ in range(max(1, int(steps))):
        optimizer.zero_grad()
        info_logits, cot_logits = model(x_bc_t)
        bc_loss = torch.nn.functional.cross_entropy(info_logits, y_info_t) + torch.nn.functional.cross_entropy(
            cot_logits, y_cot_t
        )

        pref_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        if int(x_pos_t.shape[0]) > 0:
            pos_info_logits, pos_cot_logits = model(x_pos_t)
            neg_info_logits, neg_cot_logits = model(x_neg_t)
            pos_info_logprob = torch.nn.functional.log_softmax(pos_info_logits, dim=-1)
            pos_cot_logprob = torch.nn.functional.log_softmax(pos_cot_logits, dim=-1)
            neg_info_logprob = torch.nn.functional.log_softmax(neg_info_logits, dim=-1)
            neg_cot_logprob = torch.nn.functional.log_softmax(neg_cot_logits, dim=-1)
            pos_score = pos_info_logprob.gather(1, y_info_pos_t[:, None]).squeeze(1) + pos_cot_logprob.gather(
                1, y_cot_pos_t[:, None]
            ).squeeze(1)
            neg_score = neg_info_logprob.gather(1, y_info_neg_t[:, None]).squeeze(1) + neg_cot_logprob.gather(
                1, y_cot_neg_t[:, None]
            ).squeeze(1)
            pref_loss = (torch.nn.functional.softplus(-(pos_score - neg_score)) * pair_w_t).mean()

        slot_div_loss_t = torch.tensor(float(slot_diversity_loss), dtype=torch.float32, device=device)
        loss = bc_loss + (float(preference_weight) * pref_loss) + (float(slot_diversity_weight) * slot_div_loss_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        info_logits, cot_logits = model(x_bc_t)
        info_probs = torch.nn.functional.softmax(info_logits, dim=-1)
        cot_probs = torch.nn.functional.softmax(cot_logits, dim=-1)
        info_pred = torch.argmax(info_probs, dim=-1)
        cot_pred = torch.argmax(cot_probs, dim=-1)
        info_acc = float((info_pred == y_info_t).float().mean().item())
        cot_acc = float((cot_pred == y_cot_t).float().mean().item())
        eps = 1e-9
        info_nll = float((-torch.log(info_probs[torch.arange(y_info_t.shape[0]), y_info_t] + eps)).mean().item())
        cot_nll = float((-torch.log(cot_probs[torch.arange(y_cot_t.shape[0]), y_cot_t] + eps)).mean().item())

        pair_acc = 0.0
        pair_loss = 0.0
        if int(x_pos_t.shape[0]) > 0:
            pos_info_logits, pos_cot_logits = model(x_pos_t)
            neg_info_logits, neg_cot_logits = model(x_neg_t)
            pos_info_logprob = torch.nn.functional.log_softmax(pos_info_logits, dim=-1)
            pos_cot_logprob = torch.nn.functional.log_softmax(pos_cot_logits, dim=-1)
            neg_info_logprob = torch.nn.functional.log_softmax(neg_info_logits, dim=-1)
            neg_cot_logprob = torch.nn.functional.log_softmax(neg_cot_logits, dim=-1)
            pos_score = pos_info_logprob.gather(1, y_info_pos_t[:, None]).squeeze(1) + pos_cot_logprob.gather(
                1, y_cot_pos_t[:, None]
            ).squeeze(1)
            neg_score = neg_info_logprob.gather(1, y_info_neg_t[:, None]).squeeze(1) + neg_cot_logprob.gather(
                1, y_cot_neg_t[:, None]
            ).squeeze(1)
            pair_acc = float(((pos_score - neg_score) > 0.0).float().mean().item())
            pair_loss = float((torch.nn.functional.softplus(-(pos_score - neg_score)) * pair_w_t).mean().item())

        trunk_linear = model.trunk[0]
        trained_model = DualHeadGovernanceModel(
            trunk_weight=trunk_linear.weight.detach().cpu().numpy().T.astype(np.float64),
            trunk_bias=trunk_linear.bias.detach().cpu().numpy().astype(np.float64),
            info_weight=model.info_head.weight.detach().cpu().numpy().T.astype(np.float64),
            info_bias=model.info_head.bias.detach().cpu().numpy().astype(np.float64),
            cot_weight=model.cot_head.weight.detach().cpu().numpy().T.astype(np.float64),
            cot_bias=model.cot_head.bias.detach().cpu().numpy().astype(np.float64),
        )

    return {
        "model": trained_model,
        "metrics": {
            "bc_info_acc": info_acc,
            "bc_info_nll": info_nll,
            "bc_cot_acc": cot_acc,
            "bc_cot_nll": cot_nll,
            "pair_acc": pair_acc,
            "pair_logistic_loss": pair_loss,
            "slot_diversity_weight": float(slot_diversity_weight),
            "slot_diversity_loss": float(slot_diversity_loss),
        },
    }


def save_training_artifact(
    path: str,
    *,
    encoder: FeatureEncoder,
    model: DualHeadGovernanceModel,
    metadata: Dict[str, Any],
) -> None:
    save_governance_bundle(path, encoder=encoder, model=model, metadata=metadata)
