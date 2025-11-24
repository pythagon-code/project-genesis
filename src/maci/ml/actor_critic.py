from .fnn import FNN
from typing import Any
import torch


class ActorCritic:
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = config["actor_critic"]
        self.stem = FNN(config["stem_fnn"])
        self.routing_actor = FNN(config["routing_actor_fnn"], end=True)
        self.message_actor = FNN(config["message_actor_fnn"], end=True)
        self.critics = [FNN(config["critic_fnn"], end=True) for _ in range(3)]


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.stem(x)
        x2 = torch.sigmoid(self.routing_actor(x1))
        x3 = self.message_actor(x1)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x5 = torch.min(self.critics[0](x4), self.critics[1](x4)).values
        x6 = self.critics[2](x4)
        return x2, x3, x5, x6