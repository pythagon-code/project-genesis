from dataclasses import dataclass
import torch


@dataclass(order=True)
class Message:
    step_to_receive: int
    sender_id: int
    sent_upward: bool
    encoding: torch.Tensor