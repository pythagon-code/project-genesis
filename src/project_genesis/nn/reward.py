import torch
from torch import nn


class Reward(nn.Module):
    def __init__(self) -> None:
        super().__init__()