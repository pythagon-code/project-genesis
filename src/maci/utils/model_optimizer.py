from collections import deque
from torch import nn, optim

from .mmap_object import MMapObject


class ModelOptimizer:
    def __init__(
        self,
        model_filename: str,
        optim_filename: str,
        model_containers: deque[nn.Module] | None,
        model_opt_containers: deque[tuple[nn.Module, optim.Optimizer]],
    ) -> None:
        self._model_containers = model_containers
        self._model_opt_containers = model_opt_containers

        self._model_state = MMapObject(self._model_opt_containers[0][0].state_dict(), model_filename)
        self._opt_state = MMapObject(self._model_opt_containers[0][1].state_dict(), optim_filename)


    def load_frozen(self) -> nn.Module:
        assert len(self._model_containers) > 0
        model = self._model_containers.pop()
        model.load_state_dict(self._model_state.load())
        return model


    def load_dirty(self) -> tuple[nn.Module, optim.Optimizer]:
        assert len(self._model_opt_containers) > 0
        model, opt = self._model_opt_containers.pop()
        model.load_state_dict(self._model_state.load())
        opt.load_state_dict(self._opt_state.load())
        return model, opt


    def unload_frozen(self, model: nn.Module) -> None:
        self._model_containers.append(model)


    def unload_dirty(self, model: nn.Module, opt: optim.Optimizer) -> None:
        self._model_state.save(model.state_dict())
        self._opt_state.save(opt.state_dict())
        self._model_opt_containers.append((model, opt))


    def flush(self):
        self._model_state.flush()
        self._opt_state.flush()


    def close(self):
        self._model_state.close()
        self._opt_state.close()