import torch
from torch import nn
from typing import Any

from .fnn import FNN


class Transformer(nn.Module):
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
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        q, k, v = self.q(x1), self.k(x1), self.v(x1)
        x2, _ = self.attention(q, k, v)
        x3 = self.out(torch.mean(x2, dim=-1))
        x4 = self.softmax(x3 + 1e-10)
        return x4


if __name__ == "__main__":
    from time import time
    from tqdm import tqdm
    from ..utils.configs import get_config

    print("hello")
    cfg = get_config("configs/6x6")["architecture"]
    tf = Transformer(cfg).to("cuda")
    print(tf)
    start = time()
    for i in tqdm(range(5000)):
        tf(torch.randn((64, 8, 500), device="cuda"))
    print(time() - start)