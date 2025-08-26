import torch
from typing import Any
from torch import nn


class Decomposition(nn.Module):
    def __init__(self, leve: int, config: dict[str, Any]) -> None:
        super().__init__()