from __future__ import annotations

from dataclasses import dataclass
import logging
from torch import Tensor
from typing import cast

from .cerebrum import Cerebrum
from ..ml.agent import Agent
from ..utils.model_optimizer import ModelOptimizer
from ..utils.replay_buffer import ReplayBuffer


class Neuron:
    @dataclass
    class SamplePiece:
        flow_idx: int
        piece: list[Tensor]


    def __init__(
        self,
        cerebrum: Cerebrum,
        agent: ModelOptimizer[Agent],
        replay_buffer: ReplayBuffer,
        src_neurons: list[Neuron],
        dest_neurons: list[Neuron],
        sample: list[Tensor],
        sample_idx: int
    ) -> None:
        self._cerebrum = cerebrum
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._src_neurons = src_neurons
        self._dest_neurons = dest_neurons
        self._sample = sample
        self._sample_idx = sample_idx
        self._sample_pieces = cast(list[self.SamplePiece], [])
        self._logger = logging.getLogger(self.__class__.__name__)


    def add_out_neighbor(self, neighbor: Neuron) -> None:
        self._dest_neurons.append(neighbor)
        neighbor._src_neurons.append(self)


    def collect_samples(self) -> None:
        self.


    def train(self) -> None:
        agent, opt = self._agent.load_dirty()
        dest_agents = [self._dest_neurons[i]._agent.load_frozen() for i in range(2)]

        for i in range(self._cerebrum.config["optimization"]["training"]["update_count"]):
            sample = self._replay_buffer.sample()
            loss = agent.compute_loss(dest_agents, self._cerebrum.environmental_buffer.target_actor,
                                      self._cerebrum.environmental_buffer.target_critic,,
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        self._agent.unload_dirty(agent, opt)


    def process_message(self, flow_idx: int, input_msg: Tensor) -> None:
        self._sample_pieces.append(self.SamplePiece(flow_idx, [

        ]))

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
            self._dest_neurons[route].process_message(flow_idx, output_msg)
