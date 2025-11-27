from typing import Any
import torch
from torch import nn

from .fnn import FNN


class ActorCritic(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = config["actor_critic"]
        self.stem = FNN(config["stem_fnn"])
        self.actors = nn.ModuleList(FNN(config["actor_fnn"], end=True) for _ in range(2))
        self.critics = nn.ModuleList(FNN(config["critic_fnn"], end=True) for _ in range(2))


    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        stem_out = self.stem(x)
        actors_out = [actor(stem_out) for actor in self.actors]
        critics_out = [self.critics[i](torch.cat([stem_out, actors_out[i]], dim=-1)) for i in range(2)]
        return actors_out, critics_out


if __name__ == "__main__":
    from time import time
    from tqdm import tqdm
    from ..utils.configs import get_config
    print("hello")
    cfg = get_config("configs/6x6")["architecture"]
    ac = ActorCritic(cfg).to("cuda")
    print(ac)
    start = time()
    for i in tqdm(range(5000)):
        ac(torch.randn((64, 1000), device="cuda"))
    print(time() - start)