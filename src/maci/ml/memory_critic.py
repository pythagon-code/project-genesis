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
    def __init__(self, config: dict[str, Any], device: str, rng: Generator) -> None:
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
        self._frozen_critic = FNN(actor_critic_config["critic_fnn"], end=True)
        for param in self._frozen_critic.parameters():
            param.requires_grad_(False)

        self._discount_rate: float = config["system"]["discount_rate"]
        self._loss_emv_factor: float = config["optimization"]["loss_emv_factor"]
        self._critic_loss_weight: float = config["optimization"]["critic_loss_weight"]

        self._actor_loss_ema = self._critic_loss_ema = 0.
        self._actor_loss_emv = self._critic_loss_emv = 1.

        self._hidden_state = tuple(
            torch.zeros(lstm_config["num_layers"], lstm_config["hidden_size"], device=device) for _ in range(2)
        )
        self._rng = rng


    def reset_hidden_state(self) -> None:
        self._hidden_state = tuple(torch.zeros_like(self._hidden_state[0]) for _ in range(2))


    def forward(
        self,
        hidden_state: tuple[torch.Tensor, torch.Tensor],
        states: torch.Tensor,
        critic_frozen: bool=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stem_out = self._stem(states)
        lstm_out, _ = self._lstm(stem_out, hidden_state)
        actor_out = self._actor(lstm_out)
        critic = self._frozen_critic if critic_frozen else self._critic
        critic_out = critic(torch.cat([lstm_out, actor_out], dim=-1))
        return lstm_out, critic_out


    def _harmonize_losses(
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
        hidden_states: tuple[torch.Tensor, torch.Tensor],
        all_states: torch.Tensor,
        cont_actions: torch.Tensor,
        disc_actions: torch.Tensor,
        rewards: torch.Tensor,
        target: ActorCritic
    ):
        self._frozen_critic.load_state_dict(self._critic.state_dict())

        lstm_out, all_q_values = self(hidden_states, all_states, critic_frozen=True)
        actor_loss = -torch.mean(all_q_values)

        with torch.no_grad():
            _, all_stable_q_values = target(hidden_states, all_states)
        next_q_values = all_stable_q_values[:-1]
        max_next_q_values = torch.max(next_q_values, dim=-1, keepdim=True).values
        target_q_values = rewards + self._discount_rate * max_next_q_values

        q_values = self._critic(torch.cat([lstm_out[:-1], cont_actions], dim=-1))
        predicted_q_values = q_values.gather(dim=-1, index=disc_actions)
        critic_loss = mse_loss(predicted_q_values, target_q_values)

        actor_loss, critic_loss = self._harmonize_losses(actor_loss, critic_loss)

        return actor_loss + self._critic_loss_weight * critic_loss


    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float=0.,
        gaussian_noise: float=0.
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            stem_out = self._stem(state)
            lstm_out, self._hidden_state = self._lstm(stem_out, self._hidden_state)
            actor_out = self._actor(stem_out)

            cont_action = actor_out + gaussian_noise * torch.randn_like(actor_out)
            if self._rng.random() < epsilon:
                disc_action = self._rng.integers(0, 2)
            else:
                critic_out = self._critic(torch.cat([lstm_out, cont_action], dim=-1))
                disc_action = torch.argmax(critic_out, dim=-1)

            return cont_action, disc_action


if __name__ == "__main__":
    torch.manual_seed(1)
    from time import time
    from tqdm import tqdm, trange
    from ..utils.config import get_config
    import numpy as np
    print("hello")
    cfg = get_config("configs/6x6")
    ac = ActorCritic(cfg, "cuda", default_rng(1)).to("cuda")
    new_state_dict = {}
    arr = {}
    ac_state_dict = ac.state_dict()
    for k, v in ac.state_dict().copy().items():
        v_arr = v.cpu().numpy()
        arr[k] = np.memmap(f".cache/params/ac-{k}.npy", dtype=np.float32, mode="r+", shape=v_arr.shape)
        new_state_dict[k] = torch.from_numpy(arr[k]).to(v.device)
    ac.load_state_dict(new_state_dict)
    print(arr)

    opt = optim.Adam(ac.parameters(), lr=1e-3)
    start = time()
    #hidden = torch.zeros(3, 32, 1000, device="cuda"), torch.zeros(3, 32, 1000, device="cuda")
    for i in trange(100):
        opt.zero_grad()
        loss = ac.compute_loss(
            tuple(torch.randn(3, 32, 1000, device="cuda") for _ in range(2)),
            torch.randn(7, 32, 500, device="cuda"),
            torch.randn(6, 32, 500, device="cuda"),
            torch.ones(6, 32, 1, dtype=torch.int32, device="cuda"),
            torch.randn(6, 32, 1, device="cuda"),
            ac
        )
        loss.backward()
        opt.step()
        print(loss)
        ac.select_action(torch.randn(1, 500, device="cuda"))

        for k, v in ac.state_dict().copy().items():
            v_arr = v.cpu().numpy()
            arr[k][:] = v_arr[:]
            new_state_dict[k] = torch.from_numpy(arr[k]).to(v.device)

        ac.load_state_dict(new_state_dict)


    print(time() - start)
