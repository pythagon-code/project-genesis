from .environmental_buffer import EnvironmentalBuffer
from .neuron import Neuron
import logging
import numpy as np

class Cerebrum:
    def __init__(
        self,
        config: dict,
        neurons: list[Neuron],
        environmental_buffer: EnvironmentalBuffer,
        rng: np.random.Generator,
        step: int
    ) -> None:
        self.config = config
        self._neurons = neurons
        self.environmental_buffer = environmental_buffer
        self.rng = rng
        self.step = step
        self._logger = logging.getLogger(self.__class__.__name__)


    def run(self) -> None:
        ...