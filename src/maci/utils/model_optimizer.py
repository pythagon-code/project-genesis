from collections import deque
from torch import nn, optim

from .mmap_object import MMapObject


class ModelOptimizer:
    def __init__(
        self,
        model_filename: str,
        optim_filename: str,
        frozen_model_containers: deque[nn.Module],
        dirty_model_containers: deque[nn.Module],
        opt_containers: deque[optim.Optimizer],
    ) -> None:
        self._frozen_model_containers = frozen_model_containers
        self._dirty_model_containers = dirty_model_containers
        self._opt_containers = opt_containers

        self._model_state = MMapObject(self._frozen_model_containers[0].state_dict(), model_filename)
        self._opt_state = MMapObject(self._opt_containers[0].state_dict(), optim_filename)


    def load_frozen(self) -> nn.Module:
        assert len(self._frozen_model_containers) > 0

        model = self._frozen_model_containers.pop()
        model.load_state_dict(self._model_state.load_object())

        return model


    def load_dirty(self) -> tuple[nn.Module, optim.Optimizer]:
        assert len(self._opt_containers) > 0

        model = self._dirty_model_containers.pop()
        model.load_state_dict(self._model_state.load_object())

        opt = self._opt_containers.pop()
        opt.load_state_dict(self._opt_state.load_object())

        return model, opt


    def unload_frozen(self, model: nn.Module) -> None:
        self._frozen_model_containers.append(model)


    def unload_dirty(self, model: nn.Module, opt: optim.Optimizer) -> None:
        self._model_state.unload_object(model.state_dict())
        self._dirty_model_containers.append(model)

        self._opt_state.unload_object(opt.state_dict())
        self._opt_containers.append(opt)


    def flush(self):
        self._model_state.flush()
        self._opt_state.flush()