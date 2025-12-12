from torch import Tensor, nn
from typing import Any

from .fnn import FNN


class Critic(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        critic_config = config["critic"]
        attention_config = critic_config["attention"]

        self.stem = FNN(critic_config["stem_fnn"])
        self.q = FNN(critic_config["qkv_fnn"])
        self.k = FNN(critic_config["qkv_fnn"])
        self.v = FNN(critic_config["qkv_fnn"])
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_config["embed_dim"],
            num_heads=attention_config["num_heads"],
        )
        self.out = FNN(critic_config["out_fnn"], end=True)


    def forward(self, states: Tensor) -> Tensor:
        stem_out = self.stem(states)
        q, k, v = self.q(stem_out), self.k(stem_out), self.v(stem_out)
        attn_out, _ = self.attention(q, k, v)
        actions = self.out(attn_out)
        return actions


    def compute_loss(self, states: Tensor, critic: Critic) -> Tensor: