import torch
from torch import optim
from torch.nn.functional import mse_loss
from tqdm import trange

from maci.ml.fnn import FNN


def test_simple_mutualism():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actors = [FNN([4, 16, 16, 4], output=True).to(device) for _ in range(2)]
    actors_opt = [optim.Adam(actor.parameters(), lr=1e-3) for actor in actors]

    for _ in trange(10000):
        state = torch.rand(64, 4, device=device)
        mid_action = actors[0](state)
        action = actors[1](mid_action)
        loss = mse_loss(action, state / torch.pi)
        loss.backward()
        actors_opt[0].step()
        actors_opt[0].zero_grad(set_to_none=True)

        actors_opt[1].zero_grad(set_to_none=True)
        mid_action = mid_action.detach()
        action = actors[1](mid_action)
        loss = mse_loss(action, mid_action / torch.pi)
        loss.backward()
        actors_opt[1].step()
        actors_opt[1].zero_grad(set_to_none=True)

        if _ % 1000 == 0:
            print(loss.item())

    state = torch.rand(64, 4, device="cuda")
    mid_action = actors[0](state)
    action = actors[1](mid_action)
    print(mse_loss(action, state / torch.pi).item())