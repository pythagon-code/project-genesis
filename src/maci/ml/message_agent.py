from __future__ import annotations

from numpy.random import Generator
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from ..utils.ema import get_ema_and_emv
from .actor import Actor
from .critic import Critic
from .fnn import FNN


class Agent(nn.Module):
    def __init__(self, config: dict[str, Any], device: str, rng: Generator) -> None:
        super().__init__()
        config = config["actor_critic"]
        actor_critic_config = config["architecture"]["actor_critic"]
        lstm_config = actor_critic_config["lstm"]

        self._stem = FNN(actor_critic_config["stem_fnn"])
        self._lstm = nn.LSTM(
            input_size=lstm_config["input_size"],
            hidden_size=lstm_config["hidden_size"],
            num_layers=lstm_config["num_layers"],
        )

        self._heads = nn.ModuleList(FNN(actor_critic_config["head_fnn"]) for _ in range(3))

        self._discount_rate: float = config["system"]["discount_rate"]
        self._loss_emv_factor: float = config["optimization"]["loss_emv_factor"]
        self._critic_loss_weight: float = config["optimization"]["critic_loss_weight"]

        self._actor_loss_ema = self._critic_loss_ema = 0.
        self._actor_loss_emv = self._critic_loss_emv = 1.

        self._hidden_states = tuple(
            torch.zeros(lstm_config["num_layers"], lstm_config["hidden_size"], device=device) for _ in range(2)
        )
        self._rng = rng


    def forward(
        self,
        hidden_states: tuple[Tensor, Tensor],
        states: Tensor,
    ) -> list[Tensor]:
        stem_out = self._stem(states)
        lstm_out, _ = self._lstm(stem_out, hidden_states)
        heads_out = [self._heads[i](lstm_out) for i in range(3)]
        return heads_out


    @staticmethod
    def _append_to_transformer_states(transformer_states: Tensor, other_states: Tensor) -> Tensor:
        other_states = other_states.flatten(end_dim=1).unsqueeze(0)
        return torch.cat([transformer_states, other_states])


    def _combine_losses(self, actor_loss: Tensor, critic_loss: Tensor) -> Tensor:
        harmonized_actor_loss = actor_loss / self._actor_loss_emv
        harmonized_critic_loss = critic_loss / self._critic_loss_emv

        self._actor_loss_ema, self._actor_loss_emv = get_ema_and_emv(
            self._actor_loss_ema,
            self._actor_loss_emv,
            actor_loss.item(),
            self._loss_emv_factor
        )
        self._critic_loss_ema, self._critic_loss_emv = get_ema_and_emv(
            self._critic_loss_ema,
            self._critic_loss_emv,
            critic_loss.item(),
            self._loss_emv_factor
        )

        return harmonized_actor_loss + self._critic_loss_weight * harmonized_critic_loss

    
    def compute_loss(
        self,
        hidden_states: tuple[Tensor, Tensor],
        states: Tensor,
        neighbors: list[Agent],
        neighbors_hidden_states: tuple[Tensor, Tensor],
        actor: Actor,
        critic: Critic,
        actor_critic_states: Tensor,
        actor_critic_actions: Tensor,
        actor_critic_rewards: Tensor,
        actor_critic_next_state_actions: Tensor,
    ) -> Tensor:
        messages = self(hidden_states, states)
        neighbors_messages = [neighbors[i](neighbors_hidden_states[i], messages[i]) for i in range(2)]
        submissions = neighbors_messages + [messages[2]]

        actor_in = [self._append_to_transformer_states(sub, actor_critic_states) for sub in submissions]
        actor_critic_states = torch.cat(actor_in, dim=1)
        actor_out = actor(actor_in)

        critic_in = torch.cat([actor_critic_states, actor_out], dim=-1)
        critic_out = critic(critic_in)

        actor_loss = -torch.mean(critic_out)

        critic_in = torch.cat([actor_critic_states, actor_critic_actions], dim=-1)
        q_values = critic(critic_in)
        with torch.no_grad():
            next_q_values = critic(actor_critic_next_state_actions)
            target_q_values = actor_critic_rewards + self._discount_rate * next_q_values

        critic_loss = mse_loss(q_values, target_q_values)

        return self._combine_losses(actor_loss, critic_loss)



if __name__ == "__main__":
    from time import time
    from tqdm import tqdm
    from ..utils.config import get_config
    print("hello")
    cfg = get_config("configs/6x6")["architecture"]
    ac = ActorCritic(cfg).to("cuda")
    print(ac)
    start = time()
    for i in tqdm(range(5000)):
        ac(torch.randn((64, 1000), device="cuda"))
    print(time() - start)