from .neuron import Neuron
from ..nn.memory import Memory
from ..nn.reward import Reward


class UnitNeuron(Neuron):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.memory = Memory()
        self.reward = Reward(level=0, config=self.config)


    def step(self):
        pass