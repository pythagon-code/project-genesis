import torch
from typing import Any
from torch import nn


class Memory(nn.Module):
    def __init__(self, level: int, config: dict[str, Any]) -> None:
        super().__init__()

        self.lstm = nn.LSTM(hidden_size=32, input_size=32, num_layers=8)