from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ModelBundle:
    prior_state_dict: Dict[str, torch.Tensor]
    boundary_state_dict: Dict[str, torch.Tensor]


def kl_divergence_with_probs(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def pairwise_logistic_loss(
    logits: torch.Tensor,
    winner_indices: torch.Tensor,
    loser_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    winner_scores = logits.gather(1, winner_indices[:, None]).squeeze(1)
    loser_scores = logits.gather(1, loser_indices[:, None]).squeeze(1)
    losses = torch.nn.functional.softplus(-(winner_scores - loser_scores))
    return (losses * weights).mean()
