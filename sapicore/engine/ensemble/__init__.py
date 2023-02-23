""" Ensembles (layers) manage neuron collections by fixing their size, extending attribute tensors,
and handling layer-aware operations such as heterogeneous parameter initialization
and normalization.

Notes
-----
Ensemble instance attributes are 1D tensors corresponding to potentially heterogeneous parameters
and states of neurons (tensor elements) whose behavior is prescribed by their parent class
(e.g., :class:`~neuron.spiking.LIF.LIFNeuron`).

Ensembles can be initialized programmatically or from a configuration YAML containing neuron parameters and their
generation method (fixed, zipped, grid, or drawn from a distribution). Forward calls invoke the parent neuron class
method.

Example
-------
Initialize an ensemble of basic analog neurons, inheriting default parameters and behavior from
:class:`~neuron.analog.AnalogNeuron`:

    >>> from sapicore.engine.ensemble.spiking import LIFEnsemble
    >>> layer = LIFEnsemble(num_units = 10)

"""
import torch
from torch import Tensor

from sapicore.engine.neuron import Neuron


class Ensemble(Neuron):
    """Ensemble base class. Provides generic implementations of tensor expansion and parameter diversification."""

    def __init__(self, num_units: int = 1, **kwargs):
        """Constructs a generic ensemble instance, inheriting attributes from :class:`~neuron.Neuron`."""
        super().__init__(**kwargs)
        self.num_units = num_units

        # expand configurable parameter tensors to size `num_units`.
        for prop in self._config_props_:
            temp = getattr(self, prop)
            setattr(self, prop, torch.zeros(self.num_units, dtype=torch.float, device=self.device) + temp)

        # expand loggable property tensor buffers to size `num_units`.
        for prop in self._loggable_props_:
            temp = getattr(self, prop)
            setattr(self, prop, torch.zeros(self.num_units, dtype=torch.float, device=self.device) + temp)

    def forward(self, data: Tensor) -> dict:
        """Processes the ensemble using the size-agnostic forward method of its neuron parent class.

        Parameters
        ----------
        data: Tensor
            External input to be processed by this ensemble.

        Returns
        -------
        dict
            Dictionary containing the numeric state tensor `voltage`, per the parent class method.

        Warning
        -------
        If population-aware operations are necessary (e.g., layer-wise normalization), those should go here.

        """
        return super().forward(data)
