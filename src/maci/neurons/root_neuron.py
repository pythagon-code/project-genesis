from .meta_neuron import MetaNeuron
from .neuron import Neuron


class RootNeuron(MetaNeuron):
    def __init__(self, **kwargs) -> None:
        self.neuron_list: list[Neuron] = []
        super().__init__(root=self, **kwargs)

        self.logger.info("RootNeuron initialized")
        self.logger.info(f"Total number of neurons: {len(self.neuron_list)}")

