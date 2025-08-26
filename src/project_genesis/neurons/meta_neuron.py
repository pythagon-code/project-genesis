from typing import Optional
from .neuron import Neuron
from .unit_neuron import UnitNeuron
from ..nn.composition import Composition
from ..nn.decomposition import Decomposition
from ..nn.memory import Memory
from ..nn.reward import Reward


class MetaNeuron(Neuron):
    def __init__(self, level: Optional[int]=None, topology: Optional[list[str]]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        del kwargs["name"]

        if level is None:
            self.level = len(self.config["topology"]) - 1
        else:
            self.level = level

        self.composition = Composition()
        self.decomposition = Decomposition()
        self.memory = Memory()
        self.reward = Reward(level=self.level, config=self.config)

        if topology is None:
            topology = self.config["topology"]

        graph = self.config["graphs"][topology[0]]

        self.children: list[Neuron] = []
        for i in range(graph["vertex_count"]):
            if self.level >= 1:
                child = MetaNeuron(
                    name=self.name + "-" + str(i + 1),
                    level=self.level - 1,
                    topology=topology[1:],
                    **kwargs
                )
            else:
                child = UnitNeuron(
                    name=self.name + "-" + str(i + 1),
                    **kwargs
                )

            child.set_parent(self)
            self.children.append(child)

        for u, v in graph["edges"]:
            self.children[u - 1].set_neighbor(self.children[v - 1])


    def step(self):
        pass