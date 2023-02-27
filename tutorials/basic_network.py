import torch

from sapicore.engine.ensemble.spiking import IZEnsemble
from sapicore.engine.synapse.STDP import STDPSynapse
from sapicore.engine.network import Network

from sapicore.pipeline import Pipeline
from sapicore.utils.seed import set_seed

import matplotlib.pyplot as plt


class BasicSimulation(Pipeline):
    """Simulate a basic feedforward network with two spiking layers connected by a plastic synapse."""

    _config_props_ = ("num_units", "steps")

    def __init__(self, num_units: int = 2, steps: int = 200, learning: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.steps = steps
        self.num_units = num_units
        self.learning = learning

        # fixes RNG seed for consistency across runs.
        set_seed(9846)

    def run(self):
        # initialize two default IZ ensembles.
        l1 = IZEnsemble(identifier="L1", num_units=self.num_units)
        l2 = IZEnsemble(identifier="L2", num_units=self.num_units, b=0.2632)

        # initialize a default STDP synapse and set connection probability to 0.8.
        syn = STDPSynapse(src_ensemble=l1, dst_ensemble=l2, weight_min=0.0)

        # connect the layers one-to-one and toggle learning on/off based on given setting.
        syn.connect("one")
        syn.weights = syn.weights * 15.0
        syn.toggle_learning(self.learning)

        # initialize network and add our components.
        network = Network()

        network.add_ensembles(l1, l2)
        network.add_synapses(syn)

        # conjure an input data tensor.
        data = torch.ones_like(l1.voltage) * 8.0

        # initialize tensors to store rolling simulation data.
        l1_results = torch.zeros((self.steps, self.num_units))
        l2_results = torch.zeros((self.steps, self.num_units))

        # run the simulation for `steps`.
        for i in range(self.steps):
            network(data)

            l1_results[i, :] = l1.voltage
            l2_results[i, :] = l2.voltage

        # alternatively, accumulate data into a nested dictionary using a list comprehension.
        # output = [network(data) for _ in range(self.steps)]

        plt.figure()
        plt.plot(l1_results)
        plt.title("First Layer Voltages")

        plt.figure()
        plt.plot(l2_results)
        plt.title("Second Layer Voltages")

        plt.show()


if __name__ == "__main__":
    BasicSimulation(num_units=2, steps=500, learning=True).run()
