from collections import deque
import numpy as np
from pathlib import Path
import torch
from torch import nn, optim


class ModelOptimizer:
    def __init__(
        self,
        model_filename: str,
        optim_filename: str,
        model_containers: deque[nn.Module],
        opt_containers: deque[optim.Optimizer],
        dtype: np.dtype
    ) -> None:
        self._model_containers = model_containers
        self._opt_containers = opt_containers

        self._model_state: dict[str, np.memmap] = {}
        self._opt_state: dict[str, np.memmap] = {}

        for k, v in model_containers[0].state_dict().items():
            if not v.requires_grad: continue
            assert k.replace(".", "").isalnum()
            param_path = Path(f"{model_filename}-{k}.npy")
            param_existed = param_path.exists()
            self._model_state[k] = np.memmap(param_path, dtype=dtype, mode="r+", shape=v.shape)
            if not param_existed:
                self._model_state[k][:] = v.detach().cpu().numpy()

        for k, v in opt_containers[0].state_dict().items():
            assert k.replace(".", "").isalnum()
            param_path = Path(f"{optim_filename}-{k}.npy")
            param_existed = param_path.exists()
            self._opt_state[k] = np.memmap(param_path, dtype=dtype, mode="r+", shape=v.shape)
            if not param_existed:
                self._opt_state[k][:] = v.detach().cpu().numpy()


    @staticmethod
    def _load_container(
        container: nn.Module | optim.Optimizer,
        state: dict[str, np.memmap]
    ) -> nn.Module | optim.Optimizer:
        state_dict = {}
        for k, v in state.items():
            state_dict[k] = torch.from_numpy(v)

        container.load_state_dict(state_dict, strict=isinstance(container, optim.Optimizer))
        return container


    def load_regular(self) -> nn.Module:
        assert len(self._model_containers) > 0
        model = self._load_container(self._model_containers.pop(), self._model_state)
        return model


    def load_dirty(self) -> tuple[nn.Module, optim.Optimizer]:
        assert len(self._opt_containers) > 0
        model = self._load_container(self._model_containers.pop(), self._model_state)
        opt = self._load_container(self._opt_containers.pop(), self._opt_state)
        return model, opt


    def unload_regular(self, model: nn.Module) -> None:
        self._model_containers.append(model)


    def unload_dirty(self, model: nn.Module, opt: optim.Optimizer) -> None:
        for k, v in model.state_dict().items():
            if not v.requires_grad: continue
            self._model_state[k][:] = v.detach().cpu().numpy()

        for k, v in opt.state_dict().items():
            self._opt_state[k][:] = v.detach().cpu().numpy()

        self._model_containers.append(model)
        self._opt_containers.append(opt)


    def flush(self) -> None:
        for v in self._model_state.values():
            v.flush()

        for v in self._opt_state.values():
            v.flush()