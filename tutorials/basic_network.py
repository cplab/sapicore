import torch

from sapicore.engine.ensemble.spiking import IZEnsemble
from sapicore.engine.synapse.STDP import STDPSynapse
from sapicore.engine.network import Network

from sapicore.pipeline import Pipeline
from sapicore.utils.integration import RungeKutta
from sapicore.utils.plotting import spike_raster
from sapicore.utils.seed import fix_random_seed

import matplotlib.pyplot as plt


# experiment with the following parameters:
INPUT = 5.0

A = 0.1
B = 0.27
C = -65.0

DELAY = 5.0
WEIGHT_MIN = -5.0
WEIGHT_MAX = 20.0
WEIGHT_MULTIPLIER = 5.0

NUM_UNITS = 2
STEPS = 250
LEARNING = False

INTEGRATOR = RungeKutta(order=4)


class BasicSimulation(Pipeline):
    """Simulate a basic feedforward network with two spiking IZ layers connected by one STDP synapse matrix."""

    _config_props_ = ("num_units", "steps")

    def __init__(self, num_units: int, steps: int, learning: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.steps = steps
        self.num_units = num_units
        self.learning = learning

        # fixes RNG seed across scientific stack libraries for consistency.
        fix_random_seed(9846)

    def run(self):
        # initialize two default IZ ensembles (with RK4 numeric approximation).
        l1 = IZEnsemble(identifier="L1", num_units=self.num_units, a=A, b=B, c=C, integrator=INTEGRATOR)
        l2 = IZEnsemble(identifier="L2", num_units=self.num_units, a=A, b=B, c=C, integrator=INTEGRATOR)

        # initialize a default excitatory STDP synapse with random weights.
        syn = STDPSynapse(
            src_ensemble=l1, dst_ensemble=l2, delay_ms=DELAY, weight_max=WEIGHT_MAX, weight_min=WEIGHT_MIN
        )

        # connect the layers all-to-all. see `engine.synapse.Synapse.connect` for additional settings.
        # you may also edit the `syn.connections` attribute directly (a matrix of shape source X destination).
        syn.connect("all")

        # multiply the random weights.
        syn.weights = syn.weights * WEIGHT_MULTIPLIER

        # toggle learning on/off based on given setting.
        syn.toggle_learning(self.learning)

        # initialize a network object and add our components.
        network = Network()

        network.add_ensembles(l1, l2)
        network.add_synapses(syn)

        # conjure an input data tensor.
        data = torch.ones_like(l1.voltage) * INPUT

        # initialize tensors to store rolling simulation data.
        l1_voltage = torch.zeros((self.steps, self.num_units))
        l2_voltage = torch.zeros((self.steps, self.num_units))

        l1_spiked = torch.zeros((self.steps, self.num_units))
        l2_spiked = torch.zeros((self.steps, self.num_units))

        # run the simulation for `steps`.
        for i in range(self.steps):
            network(data)

            l1_voltage[i, :] = l1.voltage
            l2_voltage[i, :] = l2.voltage

            l1_spiked[i, :] = l1.spiked
            l2_spiked[i, :] = l2.spiked

        # alternatively, accumulate data into a nested dictionary using a list comprehension.
        # output = [network(data) for _ in range(self.steps)]

        # plot all voltages.
        plt.figure()
        plt.plot(l1_voltage)
        plt.title("Layer I Voltages")

        plt.figure()
        plt.plot(l2_voltage)
        plt.title("Layer II Voltages")

        # plot spikes.
        plt.figure()
        spike_raster(torch.hstack((l1_spiked, l2_spiked)))
        plt.title("Presynaptic (Top Half) / Postsynaptic (Bottom Half)")

        # plot weights.
        plt.figure()
        plt.grid(False)

        plt.imshow(syn.weights, cmap=plt.cm.viridis, aspect="auto")
        plt.clim(torch.min(syn.weights), torch.max(syn.weights))

        plt.title("Synaptic Weights L1 (x) to L2 (y)")
        plt.colorbar()

        plt.show()


if __name__ == "__main__":
    BasicSimulation(num_units=NUM_UNITS, steps=STEPS, learning=LEARNING).run()
