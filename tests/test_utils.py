from collections import deque
import numpy as np
import os
import random
import torch
from torch import optim
from torch.nn.functional import mse_loss
from tqdm import trange

from maci.ml.fnn import FNN
from maci.utils.mmap_object import MMapObject
from maci.utils.model_optimizer import ModelOptimizer


def test_mmap_object():
    os.makedirs(".test/mmap_test/", exist_ok=True)

    arr1 = np.random.rand(100, 100, 100)
    arr_mmap = MMapObject(".test/mmap_test/arr.bin", arr1)
    arr2 = arr_mmap.load()
    assert (arr1 == arr2).all()

    for i in range(100):
        arr1 = np.random.rand(
            random.randint(1, 50),
            random.randint(1, 50),
            random.randint(1, 50),
            random.randint(1, 50)
        )
        arr_mmap.save(arr1)
        arr2 = arr_mmap.load()
        assert (arr1 == arr2).all()

    arr_mmap.close()
    arr_mmap = MMapObject(".test/mmap_test/arr.bin", arr1)
    arr2 = arr_mmap.load()
    assert (arr1 == arr2).all()
    arr_mmap.close()


def test_model_optimizer():
    os.makedirs(".test/model_optimizer_test/", exist_ok=True)

    fnn1 = FNN([10, 10], is_output=True)
    fnn2 = FNN([10, 10], is_output=True)
    for param in fnn2.parameters():
        param.requires_grad_(False)

    opt = optim.Adam(fnn1.parameters(), lr=0.1)
    fnn1(torch.randn(10, 10)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    model_containers = deque([fnn2], maxlen=1)
    model_opt_containers = deque([(fnn1, opt)], maxlen=1)

    model_opts = [ModelOptimizer(
        f".test/model_optimizer_test/model{i}.bin",
        f".test/model_optimizer_test/opt{i}.bin",
        model_containers,
        model_opt_containers
    ) for i in range(5)]

    for _ in trange(300, desc="ModelOptimizer test"):
        for i, model_opt in enumerate(model_opts):
            model, opt = model_opt.load_dirty()
            for param in model.parameters():
                assert param.requires_grad
            x = torch.randn(32, 10)
            loss = mse_loss(model(x), i * torch.ones_like(x))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            model_opt.unload_dirty(model, opt)

    for model_opt in model_opts:
        model_opt.close()

    model_opts = [ModelOptimizer(
        f".test/model_optimizer_test/model{i}.bin",
        f".test/model_optimizer_test/opt{i}.bin",
        model_containers,
        model_opt_containers
    ) for i in range(5)]

    for i, model_opt in enumerate(model_opts):
        model = model_opt.load_frozen()
        x = torch.randn(32, 10)
        loss = mse_loss(model(x), i * torch.ones_like(x))
        assert loss.item() < 1e-4
        model_opt.unload_frozen(model)
        model_opt.close()