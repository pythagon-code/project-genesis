from typing import cast

from .neuron import Neuron


class NeuronGrid:
    def __init__(self, grid_length: int,) -> None:
        self.neurons = [[Neuron(config,)]]


    def sample(self) -> Neuron:
        ...