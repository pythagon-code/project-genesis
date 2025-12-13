from collections import deque
import numpy as np
import os
import random
import torch
from torch import nn, optim

from maci.ml.fnn import FNN
from maci.utils.mmap_object import MMapObject
from maci.utils.model_optimizer import ModelOptimizer

def test_mmap_object():
    os.makedirs("temp/mmap_test/", exist_ok=True)
    if os.path.exists("temp/mmap_test/arr.bin"):
        os.remove("temp/mmap_test/arr.bin")

    arr1 = np.random.rand(100, 100, 100)
    arr_mmap = MMapObject(arr1, "temp/mmap_test/arr.bin")
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
    arr_mmap = MMapObject(arr1, "temp/mmap_test/arr.bin")
    arr2 = arr_mmap.load()
    assert (arr1 == arr2).all()


def test_model_optimizer():
    for i in range(5):
        if os.path.exists(f"temp/model_optimizer_test/model{i}.bin"):
            os.remove(f"temp/model_optimizer_test/opt{i}.bin")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fnn1 = FNN([10, 10], output=True).to(device)
    fnn2 = FNN([10, 10], output=True).to(device)
    for param in fnn2.parameters():
        param.requires_grad_(False)

    opt = optim.Adam(fnn1.parameters(), lr=1e-3)
    fnn1(torch.randn(10, 10, device=device)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    model_containers = deque([fnn2], maxlen=1)
    model_opt_containers = deque([(fnn1, opt)], maxlen=1)

    model_opts = [ModelOptimizer(
        f"temp/model_optimizer_test/model{i}.bin",
        f"temp/model_optimizer_test/opt{i}.bin",
        model_containers,
        model_opt_containers
    ) for i in range(5)]

    for _ in range(300):
        for i, model_opt in enumerate(model_opts):
            model, opt = model_opt.load_dirty()
            for param in model.parameters():
                assert param.requires_grad
            x = torch.randn(32, 10, device=device)
            loss = (torch.sum(model(x)) - i) ** 2
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            model_opt.unload_dirty(model, opt)

    for i, model_opt in enumerate(model_opts):
        model = model_opt.load_frozen()
        y = torch.sum(model(torch.randn(32, 10, device=device)))
        error = abs(y.item() - i)
        assert error < 0.1
        model_opt.unload_frozen(model)