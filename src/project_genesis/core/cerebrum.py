from ..neurons.root_neuron import RootNeuron
from ..configs.configs import get_config
import logging

class Cerebrum:
    def __init__(self) -> None:
        self.neuron1 = RootNeuron(name="Apollo", config=get_config("config1"))
        self.neuron2 = RootNeuron(name="Artemis", config=get_config("config1"))
        self.logger = logging.getLogger("Cerebrum")


    def run(self) -> None:
        ...