from abc import ABC
import torch
from torch import Tensor, nn
from typing import Any

from .fnn import FNN


class Transformer(ABC, nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        attention_config = config["attention"]
        self.stem = FNN(config["stem_fnn"])
        self.query = FNN(config["qkv_fnn"])
        self.key = FNN(config["qkv_fnn"])
        self.value = FNN(config["qkv_fnn"])
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_config["embed_dim"],
            num_heads=attention_config["num_heads"],
        )
        self.out = FNN(config["out_fnn"], is_output=True)


    def forward(self, x: Tensor, _: Tensor | None=None) -> Tensor:
        stem_out = self.stem(x)
        q, k, v = self.query(stem_out), self.key(stem_out), self.value(stem_out)
        attn_out, _ = self.attention(q, k, v)
        return self.out(torch.mean(attn_out, dim=0))