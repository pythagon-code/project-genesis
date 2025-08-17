import torch
from typing import Optional
from torch import nn
from .fnn import FNN

class Composition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_network = FNN([32, 32, 32])
        self.k_network = FNN([32, 32, 32])
        self.v_network = FNN([32, 32, 32])
        self.multiheaded_attn = nn.MultiheadAttention(32, 4)
        self.fnn = FNN([32, 32, 32])
        self.attn_weights: Optional[torch.Tensor] = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_network(x)
        k = self.k_network(x)
        v = self.v_network(x)
        attn_output, self.attn_weights = self.multiheaded_attn(q, k, v)
        return self.fnn(attn_output)