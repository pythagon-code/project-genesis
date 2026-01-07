from collections import deque
import logging
import numpy as np
import torch
from torch import Tensor, optim
from typing import cast

from .cerebrum import Cerebrum
from ..ml.actor import Actor
from ..ml.critic import Critic
from ..utils.replay_buffer import ReplayBuffer


class EnvironmentalBuffer:
    def __init__(
        self,
        cerebrum: Cerebrum,
        actor: Actor,
        actor_opt: optim.Optimizer,
        target_actor: Actor,
        critic: Critic,
        critic_opt: optim.Optimizer,
        target_critic: Critic,
        replay_buffer: ReplayBuffer
    ) -> None:
        self._cerebrum = cerebrum
        self._actor = actor
        self._actor_opt = actor_opt
        self.target_actor = target_actor
        self._critic = critic
        self._critic_opt = critic_opt
        self.target_critic = target_critic
        self._replay_buffer = replay_buffer
        self._messages = cast(deque[Tensor], deque(maxlen=cerebrum.config["system"]["flow"]["count"]))
        self.transformer_messages = np.empty(shape=[0])

        self._logger = logging.getLogger(self.__class__.__name__)


    def clear_messages(self) -> None:
        self._messages.clear()


    def submit_message(self, message: Tensor) -> None:
        assert len(self._messages) < self._messages.maxlen
        self._messages.append(message)


    def _train(self) -> None:
        for i in range(self._cerebrum.config["optimization"]["training"]["update_count"]):
            sample = self._replay_buffer.sample()
            actor_loss = self.target_actor.compute_loss(
                self._cerebrum.environmental_buffer.target_critic,
                torch.from_numpy(sample[0])
            )
            actor_loss.backward()
            self._actor_opt.step()
            self._actor_opt.zero_grad(set_to_none=True)


    def generate_final_action(self) -> Tensor:
        assert len(self._messages) == self._messages.maxlen

        if self._cerebrum.rng.random() < self._cerebrum.config["optimization"]["training"]["frequency"]:
            self._train()

        with torch.no_grad():
            return self.target_actor(torch.cat(list(self._messages), dim=0))