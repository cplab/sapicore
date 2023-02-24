import torch

from sapicore.engine.ensemble.spiking import IZEnsemble
from sapicore.engine.synapse.STDP import STDPSynapse
from sapicore.engine.network import Network

from sapicore.pipeline import Pipeline


class BasicSimulation(Pipeline):
    """Basic network with two spiking layers connected by a plastic synapse."""

    def __init__(self, num_units: int = 5, steps: int = 20, **kwargs):
        super().__init__(**kwargs)

        self.steps = steps
        self.num_units = num_units

    def run(self):
        # initialize two default IZ ensembles.
        l1 = IZEnsemble(identifier="L1", num_units=self.num_units)
        l2 = IZEnsemble(identifier="L2", num_units=self.num_units)

        # initialize a default STDP synapse and set connection probability to 0.5.
        syn = STDPSynapse(src_ensemble=l1, dst_ensemble=l2)
        syn.connect("prop", 0.8)

        # initialize network and add our components.
        network = Network()

        network.add_ensembles(l1, l2)
        network.add_synapses(syn)

        # conjure an input data tensor.
        data = torch.ones_like(l1.voltage) * 15.0

        # run the simulation for `steps` and print some feedback on every iteration.
        for i in range(self.steps):
            network(data)
            print(f"Step {i} - L1: {l1.voltage}")
            print(f"Step {i} - L2: {l2.voltage}")

        # run the network for another `steps`, accumulating the data using a list comprehension syntax.
        output = [network(data) for _ in range(self.steps)]
        print(output)


if __name__ == "__main__":
    BasicSimulation(num_units=5, steps=20).run()
