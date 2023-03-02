""" Leaky Integrate-and-Fire neuron model (LIF). """
import torch

from torch import tensor, Tensor
from torch.nn.functional import relu

from sapicore.engine.neuron.spiking import SpikingNeuron


class LIFNeuron(SpikingNeuron):
    """Base class for the leaky integrate-and-fire neuron model (LIF).

    Extends :class:`~engine.neuron.spiking.SpikingNeuron` by adding necessary parameters and implementing
    the LIF :meth:`~forward` method efficiently for 1D tensors of arbitrary size.
    User may implement subclasses to differentiate variants along additional dimensions.

    LIF units own the following attributes in addition to those inherited from
    :class:`~engine.neuron.spiking.SpikingNeuron`:

    Parameters
    ----------
    volt_thresh: float or Tensor
        Spike threshold (defaults to -55.0).

    volt_rest: float or Tensor
        Resting potential (defaults to -75.0).

    leak_gl: float or Tensor
        Leak conductance (defaults to 5.0).

    tau_mem: float or Tensor
        Membrane time constant (e.g., 5.0).

    tau_ref: float or Tensor
        Refractory period (e.g., 1.0).

    References
    ----------
    `LIF Tutorial <https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html>`_
        :meth:`~engine.neuron.spiking.LIF.LIFNeuron.forward` implements the algorithm described in this tutorial.

    """

    # loggable properties are identical to those of generic spiking neurons, so no overriding is necessary.
    _config_props_: tuple[str] = ("volt_thresh", "volt_rest", "leak_gl", "tau_mem", "tau_ref")

    def __init__(self, volt_thresh=-55.0, volt_rest=-75.0, leak_gl=5.0, tau_mem=5.0, tau_ref=1.0, **kwargs):
        """Instantiates a single LIF neuron with its default configurable and simulation attributes."""
        # register universal attributes inherited from `SpikingNeuron`, `Neuron`, and `Component`.
        super().__init__(**kwargs)

        # configurable attributes specific to LIF neurons, over and above `SpikingNeuron`.
        self.volt_thresh = torch.zeros(1, dtype=torch.float, device=self.device) + volt_thresh
        self.volt_rest = torch.zeros(1, dtype=torch.float, device=self.device) + volt_rest
        self.leak_gl = torch.zeros(1, dtype=torch.float, device=self.device) + leak_gl
        self.tau_mem = torch.zeros(1, dtype=torch.float, device=self.device) + tau_mem
        self.tau_ref = torch.zeros(1, dtype=torch.float, device=self.device) + tau_ref

        # simulation/state attributes specific to LIF, over and above `SpikingNeuron`.
        self.input = torch.zeros_like(self.volt_rest)
        self.spiked = torch.zeros_like(self.volt_rest)
        self.voltage = self.volt_rest.detach().clone()  # initialize membrane potential to resting potential.
        self.refractory_steps = torch.zeros_like(self.volt_rest).int()

        # difference equation of this neuron model, to be used with numeric approximation methods.
        self.equation = lambda x, data: (-(x - self.volt_rest) + (data / self.leak_gl)) / self.tau_mem

    def forward(self, data: Tensor) -> dict:
        """LIF forward method, to be used by ensemble or singleton 1D tensors.

        Parameters
        ----------
        data: Tensor
            External input to be processed by this LIF unit.

        Returns
        -------
        dict
            Dictionary containing the numeric state tensors `voltage`, `spiked`, and `input`.

        """
        # update internal representation of input current for tensorboard logging purposes.
        self.input = tensor([data.detach().clone()]) if not data.size() else data.detach().clone()

        # detect spikes in a pytorch-friendly way by thresholding the voltage attribute.
        spiked_prev = self.voltage >= self.volt_thresh

        # reset neurons that spiked in the previous iteration to their resting potential, or add zero otherwise.
        self.voltage = self.voltage + spiked_prev.int() * (self.volt_rest - self.voltage)
        self.spiked = spiked_prev.int()

        # add refractory steps for the neurons that spiked, otherwise do nothing.
        self.refractory_steps = self.refractory_steps + spiked_prev * torch.div(self.tau_ref, self.dt).int()

        # integrate input current, now that we know who spiked and updated their refractory status.
        self.voltage = self.integrate(data=data)

        # clamp membrane potential to the threshold `volt_thresh` if necessary.
        threshold_exceeded = (self.voltage >= self.volt_thresh).int()
        self.voltage = self.voltage + threshold_exceeded * (self.volt_thresh - self.voltage)

        # figure out which neurons in the tensor are currently refractory.
        refractory = self.refractory_steps > 0
        over_resting = self.voltage > self.volt_rest

        # if refractory, add difference between resting and current membrane potential. Otherwise, do nothing.
        self.voltage = self.voltage + (refractory.int() * over_resting.int()) * (self.volt_rest - self.voltage)

        # decrement refractory steps, taking care to zero out negatives.
        self.refractory_steps = relu(self.refractory_steps - 1)
        self.simulation_step += 1

        # return current state(s) of loggable attributes as a dictionary.
        return self.state()
