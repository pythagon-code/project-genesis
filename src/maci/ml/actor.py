import torch
from torch import Tensor
from typing import Any

from .critic import Critic
from .transformer import Transformer


class Actor(Transformer):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config["architecture"]["actor"])


    def compute_loss(self, states: Tensor, critic: Critic) -> Tensor:
        actions = self(states)
        q_values = critic(states, actions)
        return -torch.mean(q_values)