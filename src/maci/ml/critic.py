from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from typing import Any, override
import warnings

from .actor import Actor
from .transformer import Transformer


class Critic(Transformer):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config["architecture"]["critic"])
        self._discount_rate: float = config["system"]["discount_rate"]


    @override
    def forward(self, states: Tensor, actions: Tensor | None=None) -> Tensor:
        assert actions is not None
        actions = actions.unsqueeze(0).expand(states.shape[0], -1, -1)
        x = torch.cat([states, actions], dim=-1)
        return super().forward(x, None)


    def compute_loss(
        self,
        target_actor: Actor,
        target_self: Critic,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor
    ) -> Tensor:
        q_values = self(states, actions)

        with torch.no_grad():
            next_actions = target_actor(next_states)
            next_q_values = target_self(next_states, next_actions)
            target_q_values = rewards + self._discount_rate * next_q_values

        return mse_loss(q_values, target_q_values)