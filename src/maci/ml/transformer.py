from .fnn import FNN
import torch
from torch import nn
from typing import Any


class Transformer:
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = config["transformer"]
        attention_config = config["attention"]
        self.stem = FNN(config["stem_fnn"])
        self.q = FNN(config["qkv_fnn"])
        self.k = FNN(config["qkv_fnn"])
        self.v = FNN(config["qkv_fnn"])
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_config["embed_dim"],
            num_heads=attention_config["num_heads"],
            batch_first=True
        )
        self.out = FNN(config["out_fnn"], end=True)


    def forward(self, x) -> torch.Tensor:
        x1 = self.stem(x)
        q, k, v = self.q(x1), self.k(x1), self.v(x1)
        x2 = self.attention(q, k, v)[0]
        x3 = self.out(torch.mean(x2, dim=1))
        return x3