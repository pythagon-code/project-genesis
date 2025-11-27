from .fnn import FNN
import torch
from torch import nn
from typing import Any


class MemoryCritic(nn.Module):
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
        self.memory = FNN(config["memory_fnn"])
        self.critics = nn.ModuleList(FNN(config["critic_fnn"], end=True) for _ in range(2))


    def forward(
            self,
            x: torch.Tensor,
            hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        stem_out = self.stem(x)
        lstm_out, next_hidden = self.lstm(stem_out, hidden)
        memory_out = self.memory(lstm_out)
        critics_in = torch.cat([lstm_out, memory_out], dim=-1)
        critics_out = [critic(critics_in) for critic in self.critics]
        return next_hidden, memory_out, critics_out


if __name__ == "__main__":
    from time import time
    from tqdm import tqdm
    from ..utils.configs import get_config
    print("hello")
    cfg = get_config("configs/6x6")["architecture"]
    mc = MemoryCritic(cfg).to("cuda")
    print(mc)
    start = time()
    for i in tqdm(range(5000)):
        mc.hn = torch
        mc(torch.randn((64, 1, 1000), device="cuda"))
    print(time() - start)