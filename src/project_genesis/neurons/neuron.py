from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Any, Iterable, Optional, TYPE_CHECKING
from queue import PriorityQueue
from .message import Message

if TYPE_CHECKING:
    from .meta_neuron import MetaNeuron
    from .root_neuron import RootNeuron


class Neuron(ABC):
    neuron_list: list[Neuron] = []
    current_step: int = 0


    def __init__(self, name: str, config: dict[str, Any], root: RootNeuron) -> None:
        self.name = name
        self.config = config
        self.neuron_id = len(root.neuron_list)
        root.neuron_list.append(self)
        self.logger = logging.getLogger(name)
        self.parent: Optional[Neuron] = None
        self.siblings: list[Neuron] = []
        self.neuron_list.append(self)
        self.message_queue: PriorityQueue[Message] = PriorityQueue()


    def get_available_messages(self) -> Iterable[Message]:
        while not self.message_queue.empty() and self.message_queue.queue[0].step_to_receive == self.current_step:
            yield self.message_queue.get()


    def set_parent(self, parent: MetaNeuron) -> None:
        self.parent = parent


    def set_sibling(self, sibling: Neuron) -> None:
        self.siblings.append(sibling)
        sibling.siblings.append(self)


    @abstractmethod
    def step(self) -> None:
        pass