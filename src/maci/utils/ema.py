import torch
from typing import TypeVar

T = TypeVar("T", bound=torch.Tensor | float)


def get_ema(ema: T, data: T, factor: float) -> T:
    return ema + factor * (data - ema)


def get_ema_and_emv(ema: T, emv: T, data: T, factor: float) -> tuple[T, T]:
    var = (data - ema) ** 2
    return get_ema(ema, data, factor), get_ema(emv, var, factor)