import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from typing import Any

from .fnn import FNN


class RoutingAgent(FNN):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config["architecture"]["routing_agent"], output=True)


    def compute_loss(
        self,
        lstm_out: Tensor,
        actions: Tensor,
        target_q_values: Tensor
    ) -> Tensor:
        q_values = torch.gather(self(lstm_out), dim=1, index=actions)
        return mse_loss(q_values, target_q_values)