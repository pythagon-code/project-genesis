import torch
from torch import nn


class Fnn(nn.Module):
    def __init__(self, hidden_sizes: list[int]) -> None:
        super().__init__()

        layers = []
        for x in hidden_sizes:
            layers.append(nn.LazyLinear(x))
            layers.append(nn.LeakyReLU())

        layers.pop()

        self.fnn = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fnn(x)