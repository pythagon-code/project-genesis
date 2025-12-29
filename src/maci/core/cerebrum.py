from collections import deque
import logging
import numpy as np
import torch
from typing import Any

from .environmental_buffer import EnvironmentalBuffer
from .neuron import Neuron
from ..ml.agent import Agent


class Cerebrum:
    def __init__(
        self,
        config: dict[str, Any],
        neurons: list[Neuron],
        environmental_buffer: EnvironmentalBuffer,
        epsilon: float,
        gaussian_noise_var: float,
        polyak_factor: float,
        rng: np.random.Generator,
        dtype: torch.dtype,
        device: str,
        step: int
    ) -> None:
        self.config = config
        self.neurons = neurons
        self.environmental_buffer = environmental_buffer
        self.epsilon = epsilon
        self.gaussian_noise_var = gaussian_noise_var
        self.polyak_factor = polyak_factor
        self.rng = rng
        self.step = step
        self.agent_containers = deque([Agent(config, rng).to(dtype).to(device) for _ in range(3)])
        self.agent_opt_containers =
        self._logger = logging.getLogger(self.__class__.__name__)


    def run(self, num_steps: int) -> None:
        for _ in range(num_steps):

            self.step += 1