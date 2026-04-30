from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class JointActionSpace:
    info_values: Tuple[int, ...]
    cot_values: Tuple[int, ...]
    cot_token_budgets: Dict[int, int]

    def __post_init__(self) -> None:
        actions = [(info, cot) for info in self.info_values for cot in self.cot_values]
        object.__setattr__(self, "_actions", tuple(actions))
        object.__setattr__(self, "_index", {action: idx for idx, action in enumerate(actions)})
        missing = [cot for cot in self.cot_values if cot not in self.cot_token_budgets]
        if missing:
            raise ValueError(f"Missing cot token budgets for: {missing}")

    @property
    def size(self) -> int:
        return len(self._actions)

    @property
    def actions(self) -> Tuple[Tuple[int, int], ...]:
        return self._actions

    def action_to_index(self, info_mode: int, cot_mode: int) -> int:
        key = (int(info_mode), int(cot_mode))
        if key not in self._index:
            raise KeyError(f"Unknown joint action: {key}")
        return self._index[key]

    def index_to_action(self, index: int) -> Tuple[int, int]:
        return self._actions[int(index)]

    def action_label(self, index: int) -> str:
        info_mode, cot_mode = self.index_to_action(index)
        return f"i{info_mode}_c{cot_mode}"

    @property
    def info_size(self) -> int:
        return len(self.info_values)

    def info_to_index(self, info_mode: int) -> int:
        value = int(info_mode)
        try:
            return self.info_values.index(value)
        except ValueError as exc:
            raise KeyError(f"Unknown info action: {value}") from exc

    def index_to_info(self, index: int) -> int:
        return int(self.info_values[int(index)])

    def cot_budget_for_index(self, index: int) -> int:
        _, cot_mode = self.index_to_action(index)
        return int(self.cot_token_budgets[int(cot_mode)])

    def current_action_onehot(self, info_mode: int, cot_mode: int) -> List[float]:
        out = [0.0] * self.size
        out[self.action_to_index(info_mode, cot_mode)] = 1.0
        return out

    def current_info_onehot(self, info_mode: int) -> List[float]:
        out = [0.0] * self.info_size
        out[self.info_to_index(info_mode)] = 1.0
        return out

    def joint_indices_for_info(self, info_mode: int) -> List[int]:
        return [self.action_to_index(int(info_mode), cot_mode) for cot_mode in self.cot_values]


DEFAULT_ACTION_SPACE = JointActionSpace(
    info_values=(1, 2, 3, 4),
    cot_values=(0, 1, 2, 3, 4),
    cot_token_budgets={
        0: 0,
        1: 64,
        2: 256,
        3: 1024,
        4: 4096,
    },
)
