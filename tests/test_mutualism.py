import torch
from torch import optim
from torch.nn.functional import mse_loss
from tqdm import trange

from maci.ml.fnn import FNN
from maci.utils.ema import polyak_update

from . import device


def test_simple_mutualism():
    actors = [FNN([4, 16, 16, 4], output=True) for _ in range(2)]
    actors_opt = [optim.Adam(actor.parameters(), lr=1e-3) for actor in actors]

    loss = torch.ones(1)
    for _ in trange(500, desc="simple mutualism test"):
        state = torch.rand(64, 4)
        mid_action = actors[0](state)
        final_action = actors[1](mid_action)
        loss = mse_loss(final_action, state / torch.pi)
        loss.backward()
        actors_opt[0].step()
        actors_opt[0].zero_grad(set_to_none=True)
        actors_opt[1].zero_grad(set_to_none=True)

        mid_action = mid_action.detach()
        final_action = actors[1](mid_action)
        loss = mse_loss(final_action, mid_action / torch.pi)
        loss.backward()
        actors_opt[1].step()
        actors_opt[0].zero_grad(set_to_none=True)
        actors_opt[1].zero_grad(set_to_none=True)

    assert loss < 1e-3


def test_split_mutualism():
    mid_actors = [FNN([4, 32, 32, 4], output=True).to(device) for _ in range(2)]
    mid_actors_target = [FNN([4, 32, 32, 4], output=True).to(device) for _ in range(2)]
    for i in range(2):
        mid_actors_target[i].load_state_dict(mid_actors[i].state_dict())
        for param in mid_actors_target[i].parameters():
            param.requires_grad_(False)
    final_actor = FNN([8, 32, 32, 8], output=True).to(device)
    mid_actors_opt = [optim.Adam(actor.parameters(), lr=1e-3) for actor in mid_actors]
    final_actor_opt = optim.Adam(final_actor.parameters(), lr=1e-3)

    loss = torch.ones(1)
    for _ in trange(1000, desc="split mutualism test"):
        for i in range(2):
            state = torch.rand(64, 8, device=device)
            mid_action1 = mid_actors[0](state[:, :4])
            mid_action2 = mid_actors[1](state[:, 4:])
            final_action = final_actor(torch.cat([mid_action1, mid_action2], dim=-1))
            loss = mse_loss(final_action, state / torch.pi)
            loss.backward()
            mid_actors_opt[i].step()
            mid_actors_opt[0].zero_grad(set_to_none=True)
            mid_actors_opt[1].zero_grad(set_to_none=True)
            final_actor_opt.zero_grad(set_to_none=True)

            polyak_update(mid_actors_target[i], mid_actors[i], polyak_factor=0.05)

        with torch.no_grad():
            state = torch.rand(64, 8, device=device)
            mid_action1 = mid_actors_target[0](state[:, :4])
            mid_action2 = mid_actors_target[1](state[:, 4:])
        final_action = final_actor(torch.cat([mid_action1, mid_action2], dim=-1))
        loss = mse_loss(final_action, state / torch.pi)
        loss.backward()
        final_actor_opt.step()
        final_actor_opt.zero_grad(set_to_none=True)

    assert loss.item() < 1e-4


def test_independent_mutualism():
    mid_actors = [FNN([4, 32, 32, 8], output=True).to(device) for _ in range(2)]
    mid_actors_target = [FNN([4, 32, 32, 8], output=True).to(device) for _ in range(2)]
    for i in range(2):
        mid_actors_target[i].load_state_dict(mid_actors[i].state_dict())
        for param in mid_actors_target[i].parameters():
            param.requires_grad_(False)
    final_actor = FNN([8, 32, 32, 4], output=True).to(device)
    mid_actors_opt = [optim.Adam(actor.parameters(), lr=1e-3) for actor in mid_actors]
    final_actor_opt = optim.Adam(final_actor.parameters(), lr=1e-3)

    loss = torch.ones(1)
    for _ in trange(1000, desc="independent mutualism test"):
        for i in range(2):
            state = torch.rand(64, 4, device=device)
            mid_action = mid_actors[i](state)
            final_action = final_actor(mid_action)
            loss = mse_loss(final_action, state / torch.pi)
            loss.backward()
            mid_actors_opt[i].step()
            mid_actors_opt[i].zero_grad(set_to_none=True)
            final_actor_opt.zero_grad(set_to_none=True)

            polyak_update(mid_actors_target[i], mid_actors[i], polyak_factor=0.05)

        with torch.no_grad():
            state = torch.rand(64, 4, device=device)
            mid_action1 = mid_actors_target[0](state[:32, :])
            mid_action2 = mid_actors_target[1](state[32:, :])
        final_action = final_actor(torch.cat([mid_action1, mid_action2], dim=0))
        loss = mse_loss(final_action, state / torch.pi)
        loss.backward()
        final_actor_opt.step()
        final_actor_opt.zero_grad(set_to_none=True)

    assert loss.item() < 1e-4


def test_chained_mutualism():
    mid_actors = [FNN([4 + 12 * i, 32, 32, 16], output=True).to(device) for i in range(2)]
    mid_actors_target = [FNN([4 + 12 * i, 32, 32, 16], output=True).to(device) for i in range(3)]
    for i in range(2):
        mid_actors_target[i].load_state_dict(mid_actors[i].state_dict())
        for param in mid_actors_target[i].parameters():
            param.requires_grad_(False)
    final_actor = FNN([16, 32, 32, 4], output=True).to(device)
    final_actor_target = FNN([16, 32, 32, 4], output=True).to(device)
    final_actor_target.load_state_dict(final_actor.state_dict())
    for param in final_actor_target.parameters():
        param.requires_grad_(False)
    mid_actors_opt = [optim.Adam(actor.parameters(), lr=1e-3) for actor in mid_actors]
    final_actor_opt = optim.Adam(final_actor.parameters(), lr=1e-3)

    loss = torch.ones(1)
    for _ in trange(1000, desc="chained mutualism test"):
        state = torch.rand(64, 4, device=device)
        mid_action1 = mid_actors[0](state)
        mid_action2 = mid_actors_target[1](mid_action1)
        final_action = final_actor_target(mid_action2)
        loss = mse_loss(final_action, state / torch.pi)
        loss.backward()
        mid_actors_opt[0].step()
        mid_actors_opt[0].zero_grad(set_to_none=True)
        mid_actors_opt[1].zero_grad(set_to_none=True)
        final_actor_opt.zero_grad(set_to_none=True)

        state = torch.rand(64, 4, device=device)
        with torch.no_grad():
            mid_action1 = mid_actors_target[0](state)
        mid_action2 = mid_actors[1](mid_action1)
        final_action = final_actor_target(mid_action2)
        loss = mse_loss(final_action, state / torch.pi)
        loss.backward()
        mid_actors_opt[1].step()
        mid_actors_opt[1].zero_grad(set_to_none=True)
        final_actor_opt.zero_grad(set_to_none=True)

        for i in range(2):
            polyak_update(mid_actors_target[i], mid_actors[i], polyak_factor=0.05)

        state = torch.rand(64, 4, device=device)
        with torch.no_grad():
            mid_action1 = mid_actors_target[0](state)
            mid_action2 = mid_actors_target[1](mid_action1)
        final_action = final_actor(mid_action2)
        loss = mse_loss(final_action, state / torch.pi)
        loss.backward()
        final_actor_opt.step()
        final_actor_opt.zero_grad(set_to_none=True)

        polyak_update(final_actor_target, final_actor, polyak_factor=0.05)

    assert loss.item() < 1e-4