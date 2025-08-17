from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from queue import PriorityQueue
from typing import Any, Iterable, Optional, TYPE_CHECKING
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
        self.receivers: dict[Neuron, int] = {}
        self.message_queue: PriorityQueue[Message] = PriorityQueue()


    def send_message(self, message: Message) -> None:
        self.message_queue.put(message)


    def get_available_messages(self) -> Iterable[Message]:
        while not self.message_queue.empty() and self.message_queue.queue[0].step_to_receive == self.current_step:
            yield self.message_queue.get()


    def set_parent(self, parent: MetaNeuron) -> None:
        self.parent = parent
        self.receivers[parent] = 0


    def set_neighbor(self, neighbor: Neuron) -> None:
        assert neighbor not in self.receivers.keys() and neighbor not in neighbor.receivers.keys()

        self.receivers[neighbor] = len(self.receivers) + 1
        neighbor.receivers[self] = len(neighbor.receivers) + 1


    @abstractmethod
    def step(self) -> None:
        pass