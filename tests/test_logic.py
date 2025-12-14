from collections import deque
from random import sample
import torch
from torch import optim
from torch.nn.functional import mse_loss
from tqdm import trange

from maci.ml.fnn import FNN
from maci.utils.ema import polyak_update


def test_mutualism():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actors = [FNN([32, 32, 32], output=True).to(device) for _ in range(2)]
    actors_target = [FNN([32, 32, 32], output=True).to(device) for _ in range(2)]
    for i in range(2):
        actors_target[i].load_state_dict(actors[i].state_dict())
        for param in actors_target[i].parameters():
            param.requires_grad_(False)
    actors_opt = [optim.Adam(actors[i].parameters(), lr=0.1) for i in range(2)]

    critic = FNN([64, 32, 1], output=True).to(device)
    critic_target = FNN([64, 32, 1], output=True).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for param in critic_target.parameters():
        param.requires_grad_(False)
    critic_opt = optim.Adam(critic.parameters(), lr=0.1)

    replay_buffer = deque(maxlen=10000)

    for epoch in trange(200):
        with torch.no_grad():
            state = torch.randn(32, device="cuda")
            mid_action = actors_target[0](state)
            action = actors_target[1](mid_action)
            expected_action = 2 * state
            reward = -mse_loss(action, expected_action)
            replay_buffer.append((state, mid_action, action, reward))

        if len(replay_buffer) >= 32:
            batch = sample(replay_buffer, 32)
            states, mid_actions, actions, rewards = zip(*batch)
            states = torch.stack(states)
            mid_actions = torch.stack(mid_actions)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)

            new_mid_actions = actors[0](states)
            new_actions = actors_target[1](new_mid_actions)
            q_values = critic(torch.cat([states, new_actions], dim=-1))
            actor0_loss = -torch.mean(q_values)
            actor0_loss.backward()
            actors_opt[0].step()
            actors_opt[0].zero_grad(set_to_none=True)

            new_actions = actors_target[1](mid_actions)
            q_values = critic(torch.cat([states, new_actions], dim=-1))
            actor1_loss = -torch.mean(q_values)
            actor1_loss.backward()
            actors_opt[1].step()
            actors_opt[1].zero_grad(set_to_none=True)

            state_actions = torch.cat([states, actions], dim=-1)
            q_values = critic(state_actions)
            critic_loss = mse_loss(q_values, rewards)
            critic_loss.backward()
            critic_opt.step()
            critic_opt.zero_grad(set_to_none=True)

            for i in range(2):
                polyak_update(actors[i], actors_target[i], 0.95)
            polyak_update(critic, critic_target, 0.95)

    states = torch.randn(128, 32, device="cuda")
    mid_actions = actors[0](states)
    actions = actors[1](mid_actions)
    print(torch.mean(actions / states))