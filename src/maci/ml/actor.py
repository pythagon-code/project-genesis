from __future__ import annotations

import torch
from torch import TYPE_CHECKING, Tensor
from typing import Any

from .transformer import Transformer

if TYPE_CHECKING:
    from .critic import Critic


class Actor(Transformer):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config["architecture"]["actor"])


    def compute_loss(self, states: Tensor, critic: Critic) -> Tensor:
        actions = self(states)
        q_values = critic(states, actions)
        return -torch.mean(q_values)