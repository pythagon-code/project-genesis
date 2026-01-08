from __future__ import annotations

from enum import Enum
import numpy as np
from typing import Any, cast
import torch
from sympy import sequence
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from ..utils.ema import get_ema_and_emv
from .actor import Actor
from .critic import Critic
from .fnn import FNN


class Agent(nn.Module):
    def __init__(self, config: dict[str, Any], rng: np.random.Generator) -> None:
        super().__init__()

        agent_config = config["architecture"]["agent"]
        lstm_config = agent_config["lstm"]

        self._stem = FNN(agent_config["stem_fnn"])
        self._rnn = nn.RNN(
            input_size=lstm_config["input_size"],
            hidden_size=lstm_config["hidden_size"],
            num_layers=lstm_config["num_layers"],
        )
        self._message_heads = nn.ModuleList(FNN(agent_config["message_head_fnn"], is_output=True) for _ in range(3))
        self._routing_head = FNN(agent_config["routing_head_fnn"], is_output=True)

        self._discount_rate = cast(float, config["system"]["discount_rate"])
        self._loss_emv_factor = cast(float, config["optimization"]["loss_emv_factor"])
        self._actor_loss_weight = cast(float, config["optimization"]["loss_weight"]["actor"])
        self._critic_loss_weight = cast(float, config["optimization"]["loss_weight"]["critic"])
        self._routing_loss_weight = cast(float, config["optimization"]["loss_weight"]["routing"])
        self._max_abs_z = cast(float, config["system"]["gaussian_noise"]["max_abs_z"])

        self._actor_loss_ema = self._critic_loss_ema = self._routing_loss_ema = 0.
        self._actor_loss_emv = self._critic_loss_emv = self._routing_loss_emv = 1.

        self.hidden_state = torch.zeros(lstm_config["num_layers"], lstm_config["hidden_size"])
        self._rng = rng


    class RoutingMode(Enum):
        BEST_ROUTE = 0
        RANDOM_ROUTE = 1
        SUBMISSION_ROUTE = 2
        ALL_ROUTES = 3


    def forward(
        self,
        hidden_states: Tensor,
        states: Tensor,
        routing_mode: RoutingMode,
    ) -> tuple[Tensor, int, Tensor] | tuple[list[Tensor], Tensor]:
        stem_out = self._stem(states)
        rnn_out, next_hidden_states = self._rnn(stem_out, hidden_states)
        if routing_mode == self.RoutingMode.BEST_ROUTE:
            routing_q_values = self._routing_head(rnn_out)
            route = torch.argmax(routing_q_values).item()
        elif routing_mode == self.RoutingMode.RANDOM_ROUTE:
            route = self._rng.integers(low=0, high=3)
        elif routing_mode == self.RoutingMode.SUBMISSION_ROUTE:
            route = 2
        else:
            routing_q_values = self._routing_head(rnn_out)
            messages = [self._message_heads[i](rnn_out) for i in range(3)]
            return messages, routing_q_values
        message = self._message_heads[route](rnn_out)
        return message, route, next_hidden_states


    def select_action(
        self,
        hidden_state: Tensor,
        state: Tensor,
        epsilon: float,
        gaussian_noise_var: float
    ) -> tuple[Tensor, int, Tensor]:
        with torch.no_grad():
            output_msg, route, next_hidden_state = self(
                hidden_state,
                state,
                self.RoutingMode.RANDOM_ROUTE if self._rng.random() < epsilon else self.RoutingMode.BEST_ROUTE
            )
            noise_z = torch.from_numpy(
                self._rng.standard_normal(output_msg.shape),
            ).clamp(-self._max_abs_z, self._max_abs_z).to(output_msg.dtype).to(output_msg.device)
            output_msg += gaussian_noise_var * noise_z
            return output_msg, route, next_hidden_state


    def _combine_losses(
        self,
        actor_loss: Tensor,
        critic_loss: Tensor,
        routing_loss: Tensor
    ) -> Tensor:
        harmonized_actor_loss = actor_loss / self._actor_loss_emv
        harmonized_critic_loss = critic_loss / self._critic_loss_emv
        harmonized_routing_loss = routing_loss / self._routing_loss_emv

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
        self._routing_loss_ema, self._routing_loss_emv = get_ema_and_emv(
            self._routing_loss_ema,
            self._routing_loss_emv,
            routing_loss.item(),
            self._loss_emv_factor
        )

        return (
            self._actor_loss_weight * harmonized_actor_loss +
            self._critic_loss_weight * harmonized_critic_loss +
            self._routing_loss_weight * harmonized_routing_loss
        )


    # @staticmethod
    # def _append_to_transformer_states(transformer_states: Tensor, other_states: Tensor) -> Tensor:
    #     transformer_states = transformer_states.flatten(start_dim=1, end_dim=2)
    #     return torch.cat([transformer_states, other_states], dim=0)

    
    def compute_loss(
        self,
        dest_agents: list[Agent],
        target_actor: Actor,
        target_critic: Critic,
        hidden_states: Tensor,
        dest_neurons_hidden_states: Tensor,
        states: Tensor,
        transformer_other_states: Tensor,
        transformer_actions: Tensor,
        transformer_rewards: Tensor,
        transformer_next_states: Tensor,
        routes: Tensor,
    ) -> Tensor:
        messages, routing_q_values = self(hidden_states, states, self.RoutingMode.ALL_ROUTES)
        messages[2] = torch.flatten(messages[2], end_dim=1).unsqueeze(dim=0)
        dest_neurons_submissions = [dest_agents[i](
            dest_neurons_hidden_states[i].flatten(start_dim=1, end_dim=2),
            messages[i].flatten(end_dim=1).unsqueeze(dim=0),
            self.RoutingMode.SUBMISSION_ROUTE,
        )[0] for i in range(2)]
        submissions = torch.stack([messages[2]] + dest_neurons_submissions, dim=1)

        transformer_other_states = transformer_other_states.flatten(start_dim=1, end_dim=2)
        transformer_other_states = transformer_other_states.unsqueeze(dim=1).expand(-1, 3, -1, -1)
        transformer_states = torch.cat([transformer_other_states, submissions])
        transformer_states = transformer_states.flatten(start_dim=1, end_dim=2)
        transformer_new_actions = target_actor(transformer_states)
        q_values = target_critic(transformer_states, transformer_new_actions)

        actor_loss = -torch.mean(q_values)

        transformer_actions = transformer_actions.expand(3, -1, -1, -1).flatten(end_dim=2)
        q_values = target_critic(transformer_states, transformer_actions)
        with torch.no_grad():
            transformer_next_states = transformer_next_states.flatten(start_dim=1, end_dim=2)
            transformer_rewards = transformer_rewards.flatten(end_dim=1)
            transformer_next_actions = target_actor(transformer_next_states)
            next_q_values = target_critic(transformer_next_states, transformer_next_actions)
            target_q_values = transformer_rewards + self._discount_rate * next_q_values

        critic_loss = mse_loss(q_values, target_q_values.expand(3, -1, -1).flatten(end_dim=1))

        routes = routes.to(dtype=torch.int32)
        interesting_q_values = torch.gather(routing_q_values, dim=-1, index=routes).flatten(end_dim=1)

        routing_loss = mse_loss(interesting_q_values, target_q_values)

        return self._combine_losses(actor_loss, critic_loss, routing_loss)


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
    agent = Agent(cfg, default_rng(1)).to("cuda")
    dest1 = Agent(cfg, default_rng(1)).to("cuda")
    dest2 = Agent(cfg, default_rng(1)).to("cuda")
    actor = Actor(cfg).to("cuda")
    critic = Critic(cfg).to("cuda")
    for param in dest1.parameters():
        param.requires_grad_(False)
    for param in dest2.parameters():
        param.requires_grad_(False)

    agent.hidden_state = agent.hidden_state.to(torch.float32).to("cuda")

    opt = optim.Adam(agent.parameters(), lr=1e-3)
    print(agent)
    start = time()

    hidden_state = torch.randn(4, 4, device="cuda")

    for i in trange(500):
        agent.compute_loss(
            [dest1, dest2],
            actor,
            critic,
            torch.randn(4, 16, 4, device="cuda"),
            torch.randn(2, 4, 7, 16, 4, device="cuda"),
            torch.randn(7, 16, 2, device="cuda"),
            torch.randn(11, 7, 16, 2, device="cuda"),
            torch.randn(7, 16, 2, device="cuda"),
            torch.randn(7, 16, 1, device="cuda"),
            torch.randn(12, 7, 16, 2, device="cuda"),
            torch.randint(0, 3, (7, 16, 1), dtype=torch.int32, device="cuda")
        ).backward()
        agent.select_action(hidden_state, torch.randn(1, 2, device="cuda"), 0.2, 0.2)
        opt.step()
        opt.zero_grad(set_to_none=True)

    # model_opt = ModelOptimizer("dev/test/mod", "dev/test/opt", deque([agent]), deque([opt]), float32)


    print(time() - start)