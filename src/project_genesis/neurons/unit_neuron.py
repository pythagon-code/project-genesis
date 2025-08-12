from .neuron import Neuron


class UnitNeuron(Neuron):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    def step(self):
        pass