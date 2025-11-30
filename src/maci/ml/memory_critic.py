from __future__ import annotations

from math import sqrt
from numpy.random import Generator, default_rng
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from typing import Any

from ..utils.ema import get_ema
from .fnn import FNN


class ActorCritic(nn.Module):
    def __init__(self, config: dict[str, Any], rng: Generator) -> None:
        super().__init__()

        actor_critic_config = config["architecture"]["actor_critic"]
        lstm_config = actor_critic_config["lstm"]

        self._stem = FNN(actor_critic_config["stem_fnn"])
        self._lstm = nn.LSTM(
            input_size=lstm_config["input_size"],
            hidden_size=lstm_config["hidden_size"],
            num_layers=lstm_config["num_layers"],
        )
        self._actor = FNN(actor_critic_config["actor_fnn"], end=True)
        self._critic = FNN(actor_critic_config["critic_fnn"], end=True)

        self._discount_rate: float = config["system"]["discount_rate"]
        self._loss_emv_factor: float = config["optimization"]["loss_emv_factor"]
        self._critic_loss_weight: float = config["optimization"]["critic_loss_weight"]

        self._actor_loss_ema = self._critic_loss_ema = 0.
        self._actor_loss_emv = self._critic_loss_emv = 1.

        self._rng = rng


    def forward(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        stem_out = self._stem(states)
        lstm_out, next_hidden = self._lstm(stem_out, hidden)
        actor_out = self._actor(lstm_out)
        critics_out = self._critic(torch.cat([lstm_out, actions], dim=-1))
        return next_hidden, actor_out, critics_out


    def _standardize_losses(
        self,
        actor_loss: torch.Tensor,
        critic_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._actor_loss_ema = get_ema(self._actor_loss_ema, actor_loss.item(), self._loss_emv_factor)
        actor_loss_sqr_diff = (actor_loss.item() - self._actor_loss_ema) ** 2
        self._actor_loss_emv = get_ema(self._actor_loss_emv, actor_loss_sqr_diff, self._loss_emv_factor)
        actor_loss /= sqrt(self._actor_loss_emv)

        self._critic_loss_ema = get_ema(self._critic_loss_ema, critic_loss.item(), self._loss_emv_factor)
        critic_loss_sqr_diff = (critic_loss.item() - self._critic_loss_ema) ** 2
        self._critic_loss_emv = get_ema(self._critic_loss_emv, critic_loss_sqr_diff, self._loss_emv_factor)
        critic_loss /= sqrt(self._critic_loss_emv)

        return actor_loss, critic_loss


    def compute_loss(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        all_states: torch.Tensor,
        cont_actions: torch.Tensor,
        disc_actions: torch.Tensor,
        rewards: torch.Tensor,
        target: ActorCritic
    ):
        states, next_states = all_states[:-1], all_states[1:]
        _, actor_out, critic_out = self(hidden, states, cont_actions)
        actor_out = torch.cat([actor_out, torch.empty_like(actor_out[[0]])])
        _, _, all_q_values = target(hidden, all_states, actor_out)
        q_values, next_q_values = all_q_values[:-1], all_q_values[1:]

        actor_loss = -torch.mean(q_values)

        max_next_q_values = torch.max(next_q_values, dim=-1, keepdim=True).values.detach()
        target_q_values = rewards + self._discount_rate * max_next_q_values
        predicted_q_values = critic_out.gather(dim=-1, index=disc_actions)
        critic_loss = mse_loss(predicted_q_values, target_q_values)

        actor_loss, critic_loss = self._standardize_losses(actor_loss, critic_loss)

        return actor_loss + self._critic_loss_weight * critic_loss


    def select_action(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        state: torch.Tensor,
        epsilon: float=0.,
        gaussian_noise: float=0.
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stem_out = self._stem(state)
        lstm_out, _ = self._lstm(stem_out, hidden)
        actor_out = self._actor(stem_out)

        cont_action = actor_out + gaussian_noise * torch.randn_like(actor_out)
        if self._rng.random() < epsilon:
            disc_action = self._rng.integers(0, 2)
        else:
            critic_out = self._critic(torch.cat([lstm_out, cont_action], dim=-1))
            disc_action = torch.argmax(critic_out, dim=-1)

        return cont_action, disc_action


if __name__ == "__main__":
    from time import time
    from tqdm import tqdm
    from ..utils.config import get_config
    print("hello")
    cfg = get_config("configs/6x6")
    ac = ActorCritic(cfg, default_rng(1)).to("cuda")
    opt = optim.Adam(ac.parameters(), lr=1e-3)
    print(ac)
    start = time()
    hidden = torch.zeros(3, 32, 1500, device="cuda"), torch.zeros(3, 32, 1500, device="cuda")
    for i in tqdm(range(1000)):
        opt.zero_grad()

        ac.compute_loss(
            hidden,
            torch.randn(7, 32, 500, device="cuda"),
            torch.randn(6, 32, 500, device="cuda"),
            torch.ones(6, 32, 1, dtype=torch.long, device="cuda"),
            torch.randn(6, 32, 1, device="cuda"),
            ac
        ).backward()

        opt.step()

    print(time() - start)
