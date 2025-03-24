""" Ensembles (layers) manage neuron collections by fixing their size, extending select attribute tensors,
and handling layer-aware operations such as heterogeneous parameter initialization and normalization.

Note
----
Ensemble instance attributes are 1D tensors corresponding to potentially heterogeneous parameters
and states of neurons (tensor elements) whose behavior is prescribed by their parent class
(e.g., :class:`~engine.neuron.spiking.LIF.LIFNeuron`).

Ensembles can be initialized programmatically or from a configuration YAML containing neuron parameters and their
generation method (fixed, zipped, grid, or drawn from a distribution). Forward calls invoke the parent neuron class
method.

Example
-------
Initialize an ensemble of basic analog neurons, inheriting default parameters and behavior from
:class:`~engine.neuron.analog.AnalogNeuron`:

    >>> from sapicore.engine.ensemble.spiking import LIFEnsemble
    >>> layer = LIFEnsemble(num_units = 10)

See Also
--------
:class:`~utils.sweep.Sweep`
    For built-in parameter sweep utility methods.

"""
import torch
from torch import Tensor

from sapicore.engine.neuron import Neuron


class Ensemble(Neuron):
    """Ensemble base class. Provides generic implementations of tensor expansion and parameter diversification.

    Attributes
    ----------
    _extensible_props_: tuple[str]
        The subset of this object's configurable property names (`_config_props_`) that needs to be duplicated
        `num_units` times. Defaults to `None`, in which case all `_config_props_` are assumed extensible.
        In the rare case where no configurable properties should be extended, use an empty tuple.

    """

    _extensible_props_: tuple[str] = None

    def __init__(self, num_units: int = 1, **kwargs):
        """Constructs a generic ensemble instance, inheriting attributes from :class:`~engine.neuron.Neuron`."""
        super().__init__(**kwargs)
        self._num_units = num_units

        # expand extensible parameter tensors to size `num_units`.
        for prop in self.extensible_props:
            temp = getattr(self, prop)
            if temp is not None:
                setattr(self, prop, torch.zeros(self.num_units, dtype=torch.float, device=self.device) + temp)

        # expand loggable property tensor buffers to size `num_units`.
        for prop in self.loggable_props:
            temp = getattr(self, prop)
            if temp is not None:
                setattr(self, prop, torch.zeros(self.num_units, dtype=torch.float, device=self.device) + temp)

    @property
    def num_units(self):
        return self._num_units

    @property
    def extensible_props(self):
        if self._extensible_props_ is None:
            return self._config_props
        else:
            return self._extensible_props_

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
