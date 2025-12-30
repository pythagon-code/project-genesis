from __future__ import annotations

import logging
from torch import Tensor
from typing import cast

from .cerebrum import Cerebrum
from ..ml.agent import Agent
from ..utils.model_optimizer import ModelOptimizer
from ..utils.replay_buffer import ReplayBuffer


class Neuron:
    def __init__(
        self,
        neuron_id: int,
        cerebrum: Cerebrum,
        agent: ModelOptimizer[Agent],
        replay_buffer: ReplayBuffer,
        sources: list[Neuron],
        destinations: list[Neuron]
    ) -> None:
        self.neuron_id = neuron_id
        self._cerebrum = cerebrum
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._srcs = sources
        self._dests = destinations
        self._input_msgs = cast(list[Tensor], [])
        self._logger = logging.getLogger(self.__class__.__name__)


    def add_out_neighbor(self, neighbor: Neuron) -> None:
        self._dests.append(neighbor)
        neighbor._srcs.append(self)


    def receive_message(self, message: Tensor) -> None:
        self._input_msgs.append(message)


    def tick(self) -> None:
        while len(self._input_msgs) > 0:
            input_msg = self._input_msgs.pop()

            agent = self._agent.load_frozen()
            output_msg, route = agent.select_action(
                input_msg,
                self._cerebrum.epsilon,
                self._cerebrum.gaussian_noise_var
            )
            self._agent.unload_frozen(agent)

            if route == 2:
                self._cerebrum.environmental_buffer.submit_message(output_msg)
            else:
                self._dests[route].receive_message(output_msg)