from dataclasses import dataclass, field
import torch


@dataclass(order=True)
class Message:
    step_to_receive: int
    sender_id: int = field(compare=False)
    sent_upward: bool = field(compare=False)
    embedding: torch.Tensor = field(compare=False)