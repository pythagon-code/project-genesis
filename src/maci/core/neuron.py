from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
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
        piece: list[None | np.ndarray]

    def __init__(
        self,
        cerebrum: Cerebrum,
        agent: ModelOptimizer[Agent],
        hidden_state: Tensor,
        replay_buffer: ReplayBuffer,
        src_neurons: list[Neuron],
        dest_neurons: list[Neuron],
        sample: list[None | np.ndarray],
        sample_idx: int,
    ) -> None:
        self._cerebrum = cerebrum
        self._agent = agent
        self._hidden_state = hidden_state
        self._replay_buffer = replay_buffer
        self._src_neurons = src_neurons
        self._dest_neurons = dest_neurons
        self._sample = None
        self._sample_idx = 0
        self._sample_pieces = cast(list[Neuron.SamplePiece], [])
        self._next_sample_pieces = cast(list[Neuron.SamplePiece], [])
        self._logger = logging.getLogger(self.__class__.__name__)


    def add_out_neighbor(self, neighbor: Neuron) -> None:
        self._dest_neurons.append(neighbor)
        neighbor._src_neurons.append(self)


    def collect_samples(self) -> None:
        for piece in self._sample_pieces:
            transformer_state = self._cerebrum.environmental_buffer.transformer_state
            transformer_other_state = np.concat([
                transformer_state[:piece.flow_idx],
                transformer_state[piece.flow_idx + 1:]
            ])
            piece.piece[3] = transformer_other_state
            piece.piece[4] = self._cerebrum.environmental_buffer.transformer_action
            piece.piece[5] = self._cerebrum.environmental_buffer.transformer_reward
            piece.piece[6] = self._cerebrum.environmental_buffer.transformer_next_state

            self._sample[self._sample_idx] = piece.piece
            self._sample_idx += 1

            if self._sample_idx == len(self._sample):
                self._replay_buffer.push(self._sample)
                self._sample_idx = 0


    def train(self) -> None:
        agent, opt = self._agent.load_dirty()
        dest_agents = [self._dest_neurons[i]._agent.load_frozen() for i in range(2)]

        for i in range(self._cerebrum.config["optimization"]["training"]["update_count"]):
            sample = self._replay_buffer.sample()
            loss = agent.compute_loss(
                dest_agents,
                self._cerebrum.environmental_buffer.target_actor,
                self._cerebrum.environmental_buffer.target_critic,
                *sample,
            )
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        self._agent.unload_dirty(agent, opt)

    def process_message(self, flow_idx: int, input_msg: Tensor) -> None:
        agent = self._agent.load_frozen()

        output_msg, route, self._hidden_state = agent.select_action(
            self._hidden_state,
            input_msg,
            self._cerebrum.epsilon,
            self._cerebrum.gaussian_noise_var,
        )

        self._next_sample_pieces.append(self.SamplePiece(flow_idx, [
            self._hidden_state.numpy().copy(),
            input_msg.numpy().copy(),
            np.stack([self._dest_neurons[i]._hidden_state.numpy().copy() for i in range(2)]),
            None,
            None,
            None,
            None,
            np.array([route], dtype=self._cerebrum.numpy_dtype),
        ]))

        self._agent.unload_frozen(agent)

        if route == 2:
            self._cerebrum.environmental_buffer.submit_message(output_msg)
        else:
            self._dest_neurons[route].process_message(flow_idx, output_msg)
