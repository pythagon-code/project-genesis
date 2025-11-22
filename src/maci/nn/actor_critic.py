from .fnn import FNN
from typing import Any
import torch


class ActorCritic:
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.stem = FNN([1000, 1000, 1000])
        self.actor = FNN([1000, 1000, 500], True)
        self.critics = [FNN([1500, 1000, 250, 32, 1], True) for _ in range(3)]


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.stem(x)
        x2 = self.actor(x)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = torch.min(self.critics[0](x3), self.critics[1](x3)).values
        x5 = self.critics[2](x3)
        return x2, x4, x5