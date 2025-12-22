import logging
import numpy as np
from torch import Tensor, set_rng_state
from typing import Any

from .environmental_buffer import EnvironmentalBuffer
from .neuron import Neuron


class Cerebrum:
    def __init__(
        self,
        config: dict[str, Any],
        neurons: list[Neuron],
        environmental_buffer: EnvironmentalBuffer,
        epsilon: float,
        polyak_factor: float,
        rng: np.random.Generator,
        torch_rng_state: Tensor,
        step: int
    ) -> None:
        self.config = config
        self.neurons = neurons
        self.environmental_buffer = environmental_buffer
        self.epsilon = epsilon
        self.polyak_factor = polyak_factor
        self.rng = rng
        set_rng_state(torch_rng_state)
        self.step = step
        self._logger = logging.getLogger(self.__class__.__name__)


    def run(self, num_steps: int) -> None:
        while True:
            self.step += 1