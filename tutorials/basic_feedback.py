import torch

from sapicore.engine.ensemble.spiking import LIFEnsemble
from sapicore.engine.synapse.STDP import STDPSynapse
from sapicore.engine.network import Network

from sapicore.pipeline import Pipeline
from sapicore.utils.integration import RungeKutta
from sapicore.utils.seed import fix_random_seed
from sapicore.utils.plotting import spike_raster

import matplotlib.pyplot as plt

# experiment with the following parameters:
INPUT = 75.0

LEAK_GL = 2.0
TAU_MEM = 2.0
TAU_REF = 1.0

EXC_WEIGHT = 200.0
INH_WEIGHT = -200.0

DELAY_FORWARD = 5.0
DELAY_BACKWARD = 2.0

NUM_UNITS = 1
STEPS = 250

INTEGRATOR = RungeKutta(order=4)


class PING(Pipeline):
    """Simulates a minimal pyramidal-interneuron gamma compartment (PING). Experiment with the global
    variables above and observe the changes in network behavior."""

    _config_props_ = ("num_units", "steps")

    def __init__(self, num_units: int, steps: int, learning: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.steps = steps
        self.num_units = num_units
        self.learning = learning

        # fixes RNG seed across scientific stack libraries for consistency.
        fix_random_seed(9846)

    def run(self):
        # initialize two default IZ ensembles (with RK4 numeric approximation).
        l1 = LIFEnsemble(
            identifier="L1",
            num_units=self.num_units,
            integrator=INTEGRATOR,
            tau_mem=TAU_MEM,
            tau_ref=TAU_REF,
            leak_gl=LEAK_GL,
        )
        l2 = LIFEnsemble(
            identifier="L2",
            num_units=self.num_units,
            integrator=INTEGRATOR,
            tau_mem=TAU_MEM,
            tau_ref=TAU_REF,
            leak_gl=LEAK_GL,
        )

        # initialize a default excitatory STDP synapse.
        l1_l2 = STDPSynapse(src_ensemble=l1, dst_ensemble=l2, delay_ms=DELAY_FORWARD)
        l2_l1 = STDPSynapse(src_ensemble=l2, dst_ensemble=l1, delay_ms=DELAY_BACKWARD)

        l1_l2.weights = torch.ones_like(l1_l2.weights) * EXC_WEIGHT
        l2_l1.weights = torch.ones_like(l2_l1.weights) * INH_WEIGHT

        # initialize network object and add our ensembles and synapses.
        network = Network()

        network.add_ensembles(l1, l2)
        network.add_synapses(l1_l2, l2_l1)

        # connect the layers one-to-one and toggle learning on/off.
        for synapse in network.get_synapses():
            synapse.connect("one")
            synapse.set_learning(self.learning)

        # conjure an input data tensor.
        data = torch.zeros((self.steps, l1.voltage.shape[0]))
        data[int(data.shape[0] / 8) : int(data.shape[0] / 1.25), :] = INPUT

        # initialize tensors to store rolling simulation data.
        l1_voltage = torch.zeros((self.steps, self.num_units))
        l2_voltage = torch.zeros((self.steps, self.num_units))

        l1_spiked = torch.zeros((self.steps, self.num_units))
        l2_spiked = torch.zeros((self.steps, self.num_units))

        # run the simulation for `steps`.
        for i in range(self.steps):
            network(data[i, :])

            l1_voltage[i, :] = l1.voltage
            l2_voltage[i, :] = l2.voltage

            l1_spiked[i, :] = l1.spiked
            l2_spiked[i, :] = l2.spiked

        # plot voltages.
        plt.figure()
        plt.plot(l1_voltage)
        plt.plot(l2_voltage)
        plt.title("Presynaptic Excitatory (Blue) / Postsynaptic Inhibitory (Orange) Voltage Traces")
        plt.show()

        # plot spikes.
        plt.figure()
        spike_raster(torch.hstack((l1_spiked, l2_spiked)))
        plt.title("Presynaptic (Top Half) / Postsynaptic (Bottom Half)")

        plt.show()


if __name__ == "__main__":
    PING(num_units=NUM_UNITS, steps=STEPS).run()
