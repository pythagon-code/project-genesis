from __future__ import annotations
import logging
import torch

from .cerebrum import Cerebrum
from ..ml.message_agent import MessageAgent
from ..ml.routing_agent import RoutingAgent


class Neuron:
    def __init__(
        self,
        neuron_id: int,
        cerebrum: Cerebrum,
        sources: list[Neuron],
        message_agent: MessageAgent,
        routing_agent: RoutingAgent,
        destinations: list[Neuron]
    ) -> None:
        self.neuron_id = neuron_id
        self._cerebrum = cerebrum
        self._srcs = sources
        self._dests = destinations
        self._messages: list[torch.Tensor] = []
        self._message_agent = message_agent
        self._routing_agent = routing_agent
        self._logger = logging.getLogger(self.__class__.__name__)


    def add_out_neighbor(self, neighbor: Neuron) -> None:
        self._dests.append(neighbor)
        neighbor._srcs.append(self)


    def receive_message(self, message: torch.Tensor) -> None:
        self._messages.append(message)


    def tick(self) -> None:
        while len(self._messages) > 0:
            msg = self._messages.pop()
