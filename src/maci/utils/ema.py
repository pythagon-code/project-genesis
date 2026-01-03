from torch import Tensor, nn
from typing import TypeVar

T = TypeVar("T", Tensor, float)


def get_ema(ema: T, data: T, factor: float) -> T:
    return data * factor + ema * (1 - factor)


def get_ema_and_emv(ema: T, emv: T, data: T, factor: float) -> tuple[T, T]:
    var = (data - ema) ** 2
    return get_ema(ema, data, factor), get_ema(emv, var, factor)


def polyak_update(target: nn.Module, online: nn.Module, polyak_factor: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(get_ema(target_param.data, online_param.data, polyak_factor))