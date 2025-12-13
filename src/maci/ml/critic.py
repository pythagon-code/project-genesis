import torch
from torch import Tensor, nn
from typing import Any, override
import warnings

from .fnn import FNN
from .transformer import Transformer

warnings.filterwarnings("ignore")


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
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor
    ) -> Tensor:
        ...
