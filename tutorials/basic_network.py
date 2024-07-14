import torch

from sapicore.engine.ensemble.spiking import IZEnsemble
from sapicore.engine.synapse.STDP import STDPSynapse
from sapicore.engine.network import Network

from sapicore.pipeline import Pipeline
from sapicore.utils.integration import RungeKutta
from sapicore.utils.plotting import spike_raster
from sapicore.utils.seed import fix_random_seed

import matplotlib.pyplot as plt


# EXPERIMENT WITH THE FOLLOWING PARAMETERS.
INPUT = 2.0

# IZ neuron parameters.
NEURON_PARAMS = {"num_units": 2, "a": 0.1, "b": 0.27, "c": -65.0}

# STDP synapse parameters.
WEIGHT_MULTIPLIER = 1.0
STDP_PARAMS = {
    "delay_ms": 5.0,
    "weight_min": -1.0,
    "weight_max": 1.0,
    "alpha_plus": 1.0,
    "alpha_minus": 1.0,
    "tau_plus": 1.0,
    "tau_minus": 1.0,
}

# Simulation parameters.
STEPS = 500
LEARNING = True

INTEGRATOR = RungeKutta(order=4)


class BasicSimulation(Pipeline):
    """Simulate a basic feedforward network with two spiking IZ layers connected by one STDP synapse matrix."""

    _config_props_ = ("num_units", "steps")

    def __init__(self, steps: int, learning: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.steps = steps
        self.learning = learning

        # fixes RNG seed across scientific stack libraries for consistency.
        fix_random_seed(314)

    def run(self):
        # initialize two default IZ ensembles (with RK4 numeric approximation).
        l1 = IZEnsemble(identifier="L1", integrator=INTEGRATOR, **NEURON_PARAMS)
        l2 = IZEnsemble(identifier="L2", integrator=INTEGRATOR, **NEURON_PARAMS)

        # initialize a default excitatory STDP synapse with random weights.
        syn = STDPSynapse(src_ensemble=l1, dst_ensemble=l2, **STDP_PARAMS)

        # connect the layers all-to-all. see `engine.synapse.Synapse.connect` for additional settings.
        # you may also edit the `syn.connections` attribute directly (a matrix of shape source X destination).
        syn.connect("all")

        # scale the random weights to match cell parameterization.
        syn.weights = syn.weights * WEIGHT_MULTIPLIER

        # toggle learning on/off.
        syn.set_learning(self.learning)

        # initialize a network object and add our components.
        network = Network()

        network.add_ensembles(l1, l2)
        network.add_synapses(syn)

        # conjure an example input data tensor.
        data = torch.ones_like(l1.voltage) * INPUT

        # initialize tensors to store rolling simulation data.
        l1_voltage = torch.zeros((self.steps, l1.num_units))
        l2_voltage = torch.zeros((self.steps, l2.num_units))

        l1_spiked = torch.zeros((self.steps, l1.num_units))
        l2_spiked = torch.zeros((self.steps, l2.num_units))

        l1_l2_weights = torch.zeros((self.steps, l2.num_units, l1.num_units))

        # run the simulation for `steps`.
        print(str(network))
        for i in range(self.steps):
            network(data)

            l1_voltage[i, :] = l1.voltage
            l2_voltage[i, :] = l2.voltage

            l1_spiked[i, :] = l1.spiked
            l2_spiked[i, :] = l2.spiked

            l1_l2_weights[i, :, :] = syn.weights

        # alternatively, accumulate data into a nested dictionary using a list comprehension.
        # output = [network(data) for _ in range(self.steps)]

        # plot all voltages.
        plt.figure()
        plt.plot(l1_voltage.numpy())
        plt.title("Layer I Voltages")

        plt.figure()
        plt.plot(l2_voltage.numpy())
        plt.title("Layer II Voltages")

        # plot spikes.
        plt.figure()
        spike_raster(torch.hstack((l1_spiked, l2_spiked)))
        plt.title("Presynaptic (Top Half) / Postsynaptic (Bottom Half)")

        # plot weights.
        plt.figure()

        plt.title("Final Synaptic Weights L1 (x) --> L2 (y)")
        plt.imshow(syn.weights, cmap=plt.cm.viridis, aspect=1)

        plt.clim(torch.min(syn.weights), torch.max(syn.weights))
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.title("Evolution of Synaptic Weights L1 (x) --> L2 (y)")

        for i in range(l1_l2_weights.shape[1]):
            for j in range(l1_l2_weights.shape[2]):
                plt.plot(l1_l2_weights[:, i, j].numpy())

        plt.show()


if __name__ == "__main__":
    BasicSimulation(steps=STEPS, learning=LEARNING).run()
