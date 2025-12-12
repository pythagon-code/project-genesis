import torch
from torch import nn

from .fnn import FNN


class MultiFNN(nn.Module):
    def __init__(self, sizes: list[int], end: bool=False, broadcast: bool=True) -> None:
        super().__init__()
        self._fnns = nn.ModuleList(FNN(sizes, end) for _ in range(3))
        self._broadcast = broadcast


    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self._broadcast:
            return [self._fnns[i](x) for i in range(3)]
        return [self._fnns[i](x[i]) for i in range(3)]