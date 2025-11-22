from .fnn import FNN
import torch
from torch import nn
from typing import Any


class MemoryCritic:
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=1000, hidden_size=1000, num_layers=2, batch_first=True)
        self.critics = [FNN([1000, 1000, 1], True) for _ in range(2)]
        self._hn: torch.Tensor = torch.zeros(2, 1, 1000)
        self._cn: torch.Tensor = torch.zeros(2, 1000)


    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x1, (self._hn, self._cn) = self.rnn(x, (self._hn, self._cn))
        x2 = torch.min(self.critics[0](x1), self.critics[1](x1)).values
        return x1, x2