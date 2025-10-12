import torch
from torch import nn
from typing import Any, Optional
from .fnn import Fnn


class Composition(nn.Module):
    def __init__(self, level: int, config: dict[str, Any]) -> None:
        super().__init__()

        self.q_network = Fnn([4096, 4096, 4096])
        self.k_network = Fnn([4096, 4096, 4096])
        self.v_network = Fnn([4096, 4096, 4096])
        self.multiheaded_attn = nn.MultiheadAttention(4096, 4)
        self.fnn = Fnn([4096, 4096, 4096])
        self.attn_weights: Optional[torch.Tensor] = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_network(x)
        k = self.k_network(x)
        v = self.v_network(x)
        attn_output, self.attn_weights = self.multiheaded_attn(q, k, v)
        return self.fnn(attn_output)