import torch
from typing import Any
from torch import nn


class Decomposition(nn.Module):
    def __init__(self, level: int, config: dict[str, Any]) -> None:
        super().__init__()

        self.lstm = nn.LSTM()