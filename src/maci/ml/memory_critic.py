from .fnn import FNN
import torch
from torch import nn
from typing import Any


class MemoryCritic:
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = config["memory_critic"]
        lstm_config = config["lstm"]
        self.stem = FNN([1000, 1000])
        self.lstm = nn.LSTM(
            input_size=lstm_config["input_size"],
            hidden_size=lstm_config["hidden_size"],
            num_layers=lstm_config["num_layers"],
            batch_first=True
        )
        self.critics = [FNN(config["critic_fnn"], end=True) for _ in range(2)]
        self.hn: torch.Tensor = torch.zeros(1, 1, 1000)
        self.cn: torch.Tensor = torch.zeros(1, 1, 1000)


    def forward(self, x) -> torch.Tensor:
        x1 = self.stem(x)
        x2, (self.hn, self.cn) = self.lstm(x1, (self.hn, self.cn))
        x3 = torch.min(self.critics[0](x2), self.critics[1](x2)).values
        return x3