""" Izhikevich neuron model (IZ).

These neurons, while computationally efficient, also exhibit nontrivial dynamics resembling recorded
electrophysiological data (e.g., subthreshold oscillations).

Warning
-------
The results of the original Izhikevich simple model rely on :attr:`~utils.constants.DT` being set to 1.0 ms.
At finer temporal resolutions, weaker synchronization with a dominant frequency of ~25 Hz is observed.

See Also
--------
* `Original Izhikevich simple model <https://www.izhikevich.org/publications/spikes.pdf>`_ (Izhikevich, 2003).
* `Reproducibility <https://www.frontiersin.org/articles/10.3389/fninf.2018.00046/full>`_ (Pauli et al., 2018).

"""
import torch
from torch import tensor, Tensor

from sapicore.engine.neuron.spiking import SpikingNeuron


class IZNeuron(SpikingNeuron):
    """Base class for the Izhikevich neuron model.

    Extends :class:`~engine.neuron.spiking.SpikingNeuron` by adding necessary parameters and implementing
    the IZ forward method efficiently for 1D tensors of arbitrary size.

    User may implement subclasses to differentiate IZ model variants along additional dimensions
    (e.g., parameterizing ODE scalars as well).

    IZ units own the following attributes on top of those inherited from :class:`~engine.neuron.spiking.SpikingNeuron`:

    Parameters
    ----------
    volt_peak: float or Tensor
        Spike peak voltage, defaults to 35.0.

    u_curr: float or Tensor
        Recovery variable, defaults to -14.0.

    a: float or Tensor
        Timescale of the recovery variable `u_curr`, smaller is slower, defaults to 0.02.

    b: float or Tensor
        Sensitivity of the recovery variable `u_curr` to subthreshold fluctuations in membrane potential, defaults to
        0.2. Higher values couple `u_curr` and `voltage` more strongly, resulting in subthreshold oscillations and/or
        LTS-like dynamics.

    c: float or Tensor
        Post-spike reset value of the membrane potential, defaults to -65.0.

    d: float or Tensor
        Post-spike reset of the recovery variable `u_curr`, defaults to 2.0.

    Note
    ----
    Parameters are named `a`, `b`, `c`, and `d` in the original publication (Izhikevich, 2003).

    """

    _config_props_: tuple[str] = ("a", "b", "c", "d", "volt_peak")

    def __init__(self, a=0.02, b=0.2, c=-65.0, d=2.0, u_curr=-14.0, volt_peak=35.0, **kwargs):
        """Instantiates a single Izhikevich neuron with its default configurable and simulation attributes."""
        # register universal attributes inherited from `SpikingNeuron`, `Neuron`, and `Component`.
        super().__init__(**kwargs)

        # configurable attributes specific to Izhikevich neurons, over and above `SpikingNeuron`.
        self.volt_peak = torch.zeros(1, dtype=torch.float, device=self.device) + volt_peak
        self.a = torch.zeros(1, dtype=torch.float, device=self.device) + torch.as_tensor(a, device=self.device)
        self.b = torch.zeros(1, dtype=torch.float, device=self.device) + torch.as_tensor(b, device=self.device)
        self.c = torch.zeros(1, dtype=torch.float, device=self.device) + torch.as_tensor(c, device=self.device)
        self.d = torch.zeros(1, dtype=torch.float, device=self.device) + torch.as_tensor(d, device=self.device)

        # state attribute initialization.
        self.input = torch.zeros_like(self.c)
        self.spiked = torch.zeros_like(self.c)
        self.voltage = self.c.detach().clone()  # initialize membrane potential to resting potential.
        self.u_curr = torch.zeros_like(self.c) + u_curr

        # difference equations of this neuron model, to be used with numeric approximation methods.
        # note that voltage is the variable and u_curr a constant in eq. 1, and vice versa for eq. 2.
        self.equation = (
            lambda x, data, u: (0.04 * x + 5.0) * x + 140.0 - u + data,
            lambda x, v: self.a * (self.b * v - x),
        )

    def integrate(self, variable: str, **kwargs) -> Tensor:
        """Handles the multi-function integration necessary for IZ neurons (`u_curr` and `voltage`)."""
        if variable == "voltage":
            return self.integrator(x=self.voltage, equation=self.equation[0], **kwargs)

        else:
            return self.integrator(x=self.u_curr, equation=self.equation[1], **kwargs)

    def forward(self, data: Tensor) -> dict:
        """Izhikevich forward method, to be used by ensemble or singleton 1D tensors.

        Parameters
        ----------
        data: Tensor
            External input to be processed by this Izhikevich unit.

        Returns
        -------
        dict
            Dictionary containing the tensors `voltage`, `spiked`, and `input`.

        """
        # update internal representation of input current for tensorboard logging purposes.
        self.input = tensor([data.detach().clone()]) if not data.size() else data.detach().clone()

        # figure out which neurons spiked in the previous iteration.
        spike_fired = self.voltage >= self.volt_peak

        # compute membrane potential deltas for all neurons and update those that did not spike.
        self.voltage = spike_fired.int() * self.voltage + (~spike_fired).int() * self.integrate(
            "voltage", data=data, u=self.u_curr
        )

        # clamp membrane potential to threshold if now exceeds it.
        threshold_exceeded = self.voltage >= self.volt_peak
        self.voltage = self.voltage + threshold_exceeded * (self.volt_peak - self.voltage)

        # compute recovery variable deltas for all neurons and update those that did not spike.
        self.u_curr = spike_fired.int() * self.u_curr + (~spike_fired).int() * self.integrate("u_curr", v=self.voltage)

        # if threshold reached last iteration (spike_fired), record a spike, then reset `voltage` and `u_curr`.
        self.spiked = spike_fired

        self.voltage = self.voltage + spike_fired.int() * (self.c - self.voltage)
        self.u_curr = self.u_curr + spike_fired.int() * self.d

        self.simulation_step += 1

        # return current state(s) of loggable attributes as a dictionary.
        return self.loggable_state()
