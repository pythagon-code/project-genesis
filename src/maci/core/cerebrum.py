from ..utils.memory_state_buffer import MemoryStateBuffer
from .environmental_buffer import EnvironmentalBuffer
from .neuron import Neuron
import logging
import numpy as np
from typing import Any


class Cerebrum:
    def __init__(
        self,
        config: dict[str, Any],
        neurons: list[Neuron],
        environmental_buffer: EnvironmentalBuffer,
        memory_state_buffer: MemoryStateBuffer,
        rng: np.random.Generator,
        step: int
    ) -> None:
        self.config = config
        self.neurons = neurons
        self.environmental_buffer = environmental_buffer
        self.rng = rng
        self.step = step
        self._logger = logging.getLogger(self.__class__.__name__)


    def run(self) -> None:
        while True:
            self.step += 1