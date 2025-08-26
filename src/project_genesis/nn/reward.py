import torch
from typing import Any
from torch import nn
from .fnn import Fnn


class Reward(Fnn):
    def __init__(self, level: int, config: dict[str, Any]) -> None:
        super().__init__([128, 128, 128])