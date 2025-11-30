import torch
from typing import TypeVar


T = TypeVar("T", bound=torch.Tensor | float)
def get_ema(old: T, new: T, factor: float) -> T:
    return old + factor * (new - old)