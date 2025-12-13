import torch
from torch import nn


class FNN(nn.Module):
    def __init__(self, sizes: list[int], output: bool=False) -> None:
        super().__init__()

        layers = []
        prev_size = sizes[0]
        for curr_size in sizes[1:]:
            layers.append(nn.Linear(prev_size, curr_size))
            layers.append(nn.LayerNorm(curr_size))
            layers.append(nn.LeakyReLU())
            prev_size = curr_size

        if output:
            layers.pop()
            layers.pop()

        self.fc = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)