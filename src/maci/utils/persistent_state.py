from typing import Generic, TypeVar
from pathlib import Path
import torch
from torch import nn, optim

T = TypeVar("T", bound=nn.Module | optim.Optimizer)

class PersistentState(Generic[T]):
    def __init__(self, filename: str, containers: list[T], device: str) -> None:
        self._containers = containers
        self._dirty_container = containers[0]
        self._filename = Path(filename)
        self._device = device
        
        if not self._filename.exists() or not self._filename.is_file():
            self._filename.parent.mkdir(parents=True, exist_ok=True)
            self._filename.touch(exist_ok=False)
            torch.save(self._dirty_container.state_dict(), self._filename)

        self._cached_state = torch.load(self._filename, mmap=True, map_location=self._device)
        self._dirty_state = None


    def load_regular(self, container_id: int) -> T:
        assert container_id != 0 and self._dirty_state is None
        self._containers[container_id].load_state_dict(self._cached_state)
        return self._containers[container_id]


    def load_dirty(self) -> T:
        if self._dirty_state is None:
            self._dirty_state = self._cached_state
        self._dirty_container.load_state_dict(self._dirty_state)
        return self._dirty_container


    def prepare_flush(self) -> None:
        self._dirty_state = self._dirty_container.state_dict()


    def flush(self) -> None:
        assert self._dirty_state is not None
        torch.save(self._dirty_state, self._filename)
        self._cached_state = torch.load(self._filename, mmap=True, map_location=self._device)
        self._dirty_state = None