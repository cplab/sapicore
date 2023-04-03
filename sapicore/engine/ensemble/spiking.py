""" Spiking ensemble variants.

Sapicore provides leaky integrate-and-fire (LIF) and Izhikevich neuron implementations.

See Also
--------
:class:`~engine.neuron.spiking.LIF.LIFNeuron`
:class:`~engine.neuron.spiking.IZ.IZNeuron`

"""
from sapicore.engine.ensemble import Ensemble

from sapicore.engine.neuron.spiking.IZ import IZNeuron
from sapicore.engine.neuron.spiking.LIF import LIFNeuron

__all__ = ("LIFEnsemble", "IZEnsemble")


class LIFEnsemble(Ensemble, LIFNeuron):
    """Ensemble of LIF neurons, inheriting its attributes from :class:`~engine.neuron.spiking.LIF.LIFNeuron`."""

    def __init__(self, **kwargs):
        """Constructs a LIF ensemble, inheriting its attributes from :class:`~engine.neuron.spiking.LIF.LIFNeuron`."""
        super().__init__(**kwargs)

    def heterogenize(self, num_combinations: int, unravel: bool = True):
        """Ensures that calling :meth:`~heterogenize` will also update the voltage to what could be
        a changed resting potential."""

        super().heterogenize(num_combinations=self.num_units, unravel=unravel)
        self.voltage = self.volt_rest.detach().clone()


class IZEnsemble(Ensemble, IZNeuron):
    """Ensemble of IZ neurons, inheriting its attributes from :class:`~engine.neuron.spiking.IZ.IZNeuron`."""

    def __init__(self, **kwargs):
        """Constructs a LIF ensemble, inheriting its attributes from :class:`~engine.neuron.spiking.LIF.LIFNeuron`."""
        super().__init__(**kwargs)

    def heterogenize(self, num_combinations: int, unravel: bool = True):
        """Ensures that calling :meth:`~heterogenize` will also update the voltage to what could be
        a changed resting potential."""

        super().heterogenize(num_combinations=self.num_units, unravel=unravel)
        self.voltage = self.c.detach().clone()
