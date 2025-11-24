from __future__ import annotations
import logging
import torch

from .cerebrum import Cerebrum


class Neuron:
    def __init__(
        self,
        neuron_id: int,
        cerebrum: Cerebrum,
        sources: list[Neuron],
        targets: list[Neuron]
    ) -> None:
        self.neuron_id = neuron_id
        self._cerebrum = cerebrum
        self._sources = sources
        self._targets = targets
        self._messages: list[tuple[int, torch.Tensor]] = []
        self._logger = logging.getLogger(self.__class__.__name__)


    def add_out_neighbor(self, neighbor: Neuron) -> None:
        self._targets.append(neighbor)
        neighbor._sources.append(self)


    def _send_message(self, message: torch.Tensor, receiver: Neuron) -> None:
        receiver._messages.append((receiver._sources.index(self), message))


    def _submit(self, message: torch.Tensor) -> None:
        self._cerebrum.environmental_buffer.receive_message(message, self)


    def tick(self) -> None:
        ...