from __future__ import annotations
from .cerebrum import Cerebrum
import logging
import torch


class Neuron:
    def __init__(
        self,
        cerebrum: Cerebrum,
        sources: list[Neuron],
        targets: list[Neuron]
    ) -> None:
        self._cerebrum = cerebrum
        self._logger = logging.getLogger(self.__class__.__name__)
        self._sources = sources
        self._targets = targets
        self._messages: list[tuple[int, torch.Tensor]] = []


    def tick(self) -> None:
        pass