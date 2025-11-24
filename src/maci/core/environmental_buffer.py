import logging
import torch

from .cerebrum import Cerebrum
from .neuron import Neuron


class EnvironmentalBuffer:
    def __init__(
        self,
        cerebrum: Cerebrum
    ) -> None:
        self._cerebrum = cerebrum
        self._messages: list[tuple[int, torch.Tensor]] = []
        self._logger = logging.getLogger(self.__class__.__name__)


    def _send_message(self, message: torch.Tensor, receiver: Neuron) -> None:
        ...


    def receive_message(self, message: torch.Tensor, sender: Neuron) -> None:
        self._messages.append((sender.neuron_id, message))