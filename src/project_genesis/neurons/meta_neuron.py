from typing import Optional
from .neuron import Neuron
from .unit_neuron import UnitNeuron

class MetaNeuron(Neuron):
    def __init__(self, topology: Optional[list[str]]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        del kwargs["name"]

        if topology is None:
            topology = self.config["topology"]

        graph = self.config["graphs"][topology[0]]

        self.children: list[Neuron] = []
        for i in range(graph["vertex_count"]):
            if len(topology) > 1:
                child = MetaNeuron(
                    name=self.name + "-" + str(i + 1),
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
            self.children[u - 1].set_sibling(self.children[v - 1])


    def step(self):
        pass