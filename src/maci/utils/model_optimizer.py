from collections import deque
from torch import nn, optim
from typing import TypeVar

from .mmap_object import MMapObject

ModelType = TypeVar("ModelType", bound=nn.Module)


class ModelOptimizer[ModelType]:
    def __init__(
        self,
        model_filename: str,
        optim_filename: str,
        model_containers: deque[ModelType] | None,
        model_opt_containers: deque[tuple[ModelType, optim.Optimizer]],
    ) -> None:
        self._model_containers = model_containers
        self._model_opt_containers = model_opt_containers

        self._model_state = MMapObject(model_filename, self._model_opt_containers[0][0].state_dict())
        self._opt_state = MMapObject(optim_filename, self._model_opt_containers[0][1].state_dict())


    def load_frozen(self) -> ModelType:
        assert len(self._model_containers) > 0
        model = self._model_containers.pop()
        model.load_state_dict(self._model_state.load())
        return model


    def load_dirty(self) -> tuple[ModelType, optim.Optimizer]:
        assert len(self._model_opt_containers) > 0
        model, opt = self._model_opt_containers.pop()
        model.load_state_dict(self._model_state.load())
        opt.load_state_dict(self._opt_state.load())
        return model, opt


    def unload_frozen(self, model: ModelType) -> None:
        self._model_containers.append(model)


    def unload_dirty(self, model: ModelType, opt: optim.Optimizer) -> None:
        self._model_state.save(model.state_dict())
        self._opt_state.save(opt.state_dict())
        self._model_opt_containers.append((model, opt))


    def flush(self) -> None:
        self._model_state.flush()
        self._opt_state.flush()


    def close(self) -> None:
        self._model_state.close()
        self._opt_state.close()