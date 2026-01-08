from collections import deque
import logging
import numpy as np
import torch
from torch import optim
from typing import Any

from .environmental_buffer import EnvironmentalBuffer
from .neuron import Neuron
from ..ml.agent import Agent


class Cerebrum:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.neurons = [Neuron() if _ in range]
        # self.epsilon = epsilon
        # self.gaussian_noise_var = gaussian_noise_var
        # self.polyak_factor = polyak_factor
        # self.rng = rng
        # self.numpy_dtype = numpy_dtype
        # self.torch_dtype = torch_dtype
        # self.device = device
        # self.step = step
        # self.agent_containers = deque([Agent(config, rng).to(dtype).to(device) for _ in range(3)])
        # dirty_agent = Agent(config, rng).to(dtype).to(device)
        # self.agent_opt_containers = deque([dirty_agent, optim.Adam(dirty_agent.parameters(), lr=learning_rate)])
        self._logger = logging.getLogger(self.__class__.__name__)


    def __getstate__(self) -> tuple:
        return (
            self.config,
            self.neurons,
            self.environmental_buffer,
            self.epsilon,
            self.gaussian_noise_var,
            self.polyak_factor,
            self.rng,
            self.numpy_dtype,
            self.torch_dtype,
            self.device,
            self.step,
            self.frozen_agent_containers,
            self.dirty_agent_containers
        )


    def __setstate__(self, state: tuple) -> None:
        (
            self.config,
            self.neurons,
            self.environmental_buffer,
            self.epsilon,
            self.gaussian_noise_var,
            self.polyak_factor,
            self.rng,
            self.numpy_dtype,
            self.torch_dtype,
            self.device,
            self.step,
            self.frozen_agent_containers,
            self.dirty_agent_containers
        ) = state
        self._logger = logging.getLogger(self.__class__.__name__)


    def run(self, num_steps: int) -> None:
        for _ in range(num_steps):
            for neuron in self.neurons:
                neuron.tick()

            self.step += 1