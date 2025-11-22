from .fnn import FNN
import torch
from torch import nn
from typing import Any


class Transformer:
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.stem = FNN([1000, 1000])
        self.q_fc = FNN([500, 500])
        self.k_fc = FNN([500, 500])
        self.v_fc = FNN([500, 500])
        self.attention = nn.MultiheadAttention(embed_dim=500, num_heads=4, batch_first=True)
        self.fc = FNN([500, 1000, 1000])


    def forward(self, x) -> torch.Tensor:
        x1 = self.stem(x)
        q, k, v = self.q_fc(x1), self.k_fc(x1), self.v_fc(x1)
        x2 = self.attention(q, k, v)[0]
        x3 = self.fc(torch.mean(x2, dim=1))
        return x3