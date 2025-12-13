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


class MessageAgent(nn.Module):
    def __init__(self, config: dict[str, Any], device: str, rng: Generator) -> None:
        super().__init__()

        message_agent_config = config["architecture"]["message_agent"]
        lstm_config = message_agent_config["lstm"]

        self._stem = FNN(message_agent_config["stem_fnn"])
        self._lstm = nn.LSTM(
            input_size=lstm_config["input_size"],
            hidden_size=lstm_config["hidden_size"],
            num_layers=lstm_config["num_layers"],
        )
        self._heads = nn.ModuleList(FNN(message_agent_config["head_fnn"]) for _ in range(3))

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
        hidden_states: tuple[Tensor, ...],
        states: Tensor,
        which_heads: list[int] | None=None
    ) -> list[Tensor]:
        which_heads = which_heads or list(range(3))
        stem_out = self._stem(states)
        lstm_out, _ = self._lstm(stem_out, hidden_states)
        heads_out = [self._heads[i](lstm_out) for i in which_heads]
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
        hidden_states: tuple[Tensor, ...],
        states: Tensor,
        dsts: list[MessageAgent],
        dsts_hidden_states: list[tuple[Tensor, ...]],
        actor: Actor,
        critic: Critic,
        transformer_states: Tensor,
        transformer_rewards: Tensor,
        transformer_next_states: Tensor,
    ) -> Tensor:
        messages = self(hidden_states, states)
        dsts_messages = [dsts[i](dsts_hidden_states[i], messages[i], [0])[0] for i in range(2)]
        submissions = [messages[0]] + dsts_messages

        transformer_states = [self._append_to_transformer_states(transformer_states, sub) for sub in submissions]
        transformer_states = torch.cat(transformer_states, dim=1)
        transformer_actions = actor(transformer_states)

        q_values = critic(transformer_states, transformer_actions)

        actor_loss = -torch.mean(q_values)

        with torch.no_grad():
            transformer_next_actions = actor(transformer_next_states)
            next_q_values = critic(transformer_next_states, transformer_next_actions)
            target_q_values = transformer_rewards + self._discount_rate * next_q_values
            target_q_values = target_q_values.repeat_interleave(3, dim=0).squeeze(0)

        critic_loss = mse_loss(q_values, target_q_values)

        return self._combine_losses(actor_loss, critic_loss)


if __name__ == "__main__":
    from time import time
    from tqdm import trange
    from ..utils.config import get_config
    from torch import optim
    from numpy.random import default_rng
    from ..utils.model_optimizer import ModelOptimizer
    from collections import deque
    from numpy import float32
    print("hello")
    cfg = get_config("configs/6x6")
    agent = MessageAgent(cfg, "cuda", default_rng(1)).to("cuda")
    dst1 = MessageAgent(cfg, "cuda", default_rng(1)).to("cuda")
    dst2 = MessageAgent(cfg, "cuda", default_rng(1)).to("cuda")
    actor = Actor(cfg).to("cuda")
    critic = Critic(cfg).to("cuda")
    for param in dst1.parameters():
        param.requires_grad_(False)
    for param in dst2.parameters():
        param.requires_grad_(False)

    opt = optim.Adam(agent.parameters(), lr=1e-3)
    print(agent)
    start = time()
    for i in trange(100):
        agent.compute_loss(
            tuple(torch.randn(3, 16, 4, device="cuda") for _ in range(2)),
            torch.randn(7, 16, 2, device="cuda"),
            [dst1, dst2],
            [tuple(torch.randn(3, 16, 4, device="cuda") for _ in range(2)) for _ in range(2)],
            actor,
            critic,
            torch.randn(11, 7 * 16, 2, device="cuda"),
            torch.randn(7 * 16, 1, device="cuda"),
            torch.randn(12, 7 * 16, 2, device="cuda")
        ).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    model_opt = ModelOptimizer("dev/test/mod", "dev/test/opt",
                               deque([agent]), deque([opt]), float32)


    print(time() - start)