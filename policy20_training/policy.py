from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .action_space import JointActionSpace
from .features import BoundaryFeatureSpec, HashTextVectorizer
from .models import MLP
from .slot_memory import SlotMemory


def _state_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    state = row.get("state_features", row)
    return state if isinstance(state, dict) else row


@dataclass
class OnlineDecisionPolicy:
    action_space: JointActionSpace
    text_vectorizer: HashTextVectorizer
    boundary_spec: BoundaryFeatureSpec
    slot_memory: SlotMemory
    prior_model: MLP
    think_boundary_model: MLP
    answer_boundary_model: MLP
    prior_weight: float = 0.35
    boundary_weight: float = 1.0
    switch_cost: float = 0.10
    hysteresis_bonus: float = 0.05
    budget_guardrail_penalty: float = 0.35
    invalid_action_penalty: float = 1e9
    device: str = "cpu"

    def _prompt_features(self, prompt: str) -> np.ndarray:
        return self.text_vectorizer.transform_one(prompt)[None, :]

    def _slot_prior(self, prompt: str) -> np.ndarray:
        return self.slot_memory.query(self._prompt_features(prompt))

    def _boundary_features(self, boundary_row: Dict[str, Any]) -> np.ndarray:
        return self.boundary_spec.transform([boundary_row])

    def score_initial_actions(self, prompt: str) -> np.ndarray:
        prompt_features = self._prompt_features(prompt)
        slot_prior = self._slot_prior(prompt)
        x = np.concatenate([prompt_features, slot_prior], axis=1)
        with torch.no_grad():
            logits = self.prior_model(torch.tensor(x, dtype=torch.float32, device=self.device))
        return logits[0].cpu().numpy()

    def _collapse_joint_scores_to_info(self, joint_scores: np.ndarray) -> np.ndarray:
        info_scores = np.zeros((self.action_space.info_size,), dtype=np.float32)
        for info_idx, info_mode in enumerate(self.action_space.info_values):
            joint_indices = self.action_space.joint_indices_for_info(int(info_mode))
            info_scores[info_idx] = float(np.mean(joint_scores[joint_indices]))
        return info_scores

    def _expand_info_scores_to_joint(self, info_scores: np.ndarray, fixed_cot_mode: int) -> np.ndarray:
        expanded = np.full((self.action_space.size,), -float(self.invalid_action_penalty), dtype=np.float32)
        for info_idx, info_mode in enumerate(self.action_space.info_values):
            joint_idx = self.action_space.action_to_index(int(info_mode), int(fixed_cot_mode))
            expanded[joint_idx] = float(info_scores[info_idx])
        return expanded

    def _score_think_boundary_actions(
        self,
        *,
        prompt: str,
        boundary_row: Dict[str, Any],
        current_action: Tuple[int, int],
        remaining_thinking_budget_tokens: int | None,
    ) -> np.ndarray:
        prompt_features = self._prompt_features(prompt)
        slot_prior = self._slot_prior(prompt)
        boundary_features = self._boundary_features(boundary_row)
        current_onehot = np.asarray(
            [self.action_space.current_action_onehot(current_action[0], current_action[1])],
            dtype=np.float32,
        )
        x = np.concatenate([prompt_features, slot_prior, boundary_features, current_onehot], axis=1)
        with torch.no_grad():
            boundary_logits = self.think_boundary_model(
                torch.tensor(x, dtype=torch.float32, device=self.device)
            )[0].cpu().numpy()
        prior_logits = self.score_initial_actions(prompt)
        combined = (self.prior_weight * prior_logits) + (self.boundary_weight * boundary_logits)

        current_idx = self.action_space.action_to_index(current_action[0], current_action[1])
        for idx in range(self.action_space.size):
            if idx == current_idx:
                combined[idx] += float(self.hysteresis_bonus)
            else:
                combined[idx] -= float(self.switch_cost)
            if remaining_thinking_budget_tokens is not None:
                if self.action_space.cot_budget_for_index(idx) > int(remaining_thinking_budget_tokens):
                    combined[idx] -= float(self.budget_guardrail_penalty)
        return combined

    def _score_answer_boundary_actions(
        self,
        *,
        prompt: str,
        boundary_row: Dict[str, Any],
        current_action: Tuple[int, int],
    ) -> np.ndarray:
        prompt_features = self._prompt_features(prompt)
        slot_prior = self._slot_prior(prompt)
        boundary_features = self._boundary_features(boundary_row)
        current_info_onehot = np.asarray(
            [self.action_space.current_info_onehot(current_action[0])],
            dtype=np.float32,
        )
        x = np.concatenate([prompt_features, slot_prior, boundary_features, current_info_onehot], axis=1)
        with torch.no_grad():
            answer_logits = self.answer_boundary_model(
                torch.tensor(x, dtype=torch.float32, device=self.device)
            )[0].cpu().numpy()
        prior_joint_logits = self.score_initial_actions(prompt)
        prior_info_logits = self._collapse_joint_scores_to_info(prior_joint_logits)
        combined_info = (self.prior_weight * prior_info_logits) + (self.boundary_weight * answer_logits)

        current_info_idx = self.action_space.info_to_index(current_action[0])
        for info_idx in range(self.action_space.info_size):
            if info_idx == current_info_idx:
                combined_info[info_idx] += float(self.hysteresis_bonus)
            else:
                combined_info[info_idx] -= float(self.switch_cost)
        return self._expand_info_scores_to_joint(combined_info, fixed_cot_mode=int(current_action[1]))

    def score_boundary_actions(
        self,
        *,
        prompt: str,
        boundary_row: Dict[str, Any],
        current_action: Tuple[int, int],
        remaining_thinking_budget_tokens: int | None = None,
    ) -> np.ndarray:
        state = _state_payload(boundary_row)
        if bool(state.get("is_answer_zone", boundary_row.get("is_answer_zone", False))):
            return self._score_answer_boundary_actions(
                prompt=prompt,
                boundary_row=boundary_row,
                current_action=current_action,
            )
        return self._score_think_boundary_actions(
            prompt=prompt,
            boundary_row=boundary_row,
            current_action=current_action,
            remaining_thinking_budget_tokens=remaining_thinking_budget_tokens,
        )

    def choose_boundary_action(
        self,
        *,
        prompt: str,
        boundary_row: Dict[str, Any],
        current_action: Tuple[int, int],
        remaining_thinking_budget_tokens: int | None = None,
    ) -> Dict[str, Any]:
        scores = self.score_boundary_actions(
            prompt=prompt,
            boundary_row=boundary_row,
            current_action=current_action,
            remaining_thinking_budget_tokens=remaining_thinking_budget_tokens,
        )
        best_idx = int(np.argmax(scores))
        best_action = self.action_space.index_to_action(best_idx)
        return {
            "best_action_index": best_idx,
            "best_action": {"info_mode": int(best_action[0]), "cot_mode": int(best_action[1])},
            "scores": scores.tolist(),
        }
