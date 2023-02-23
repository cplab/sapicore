"""Components manage basic properties and logic common to neurons, synapses, and their derivative classes."""
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from sapicore.utils.constants import DT
from sapicore.utils.io import dump_yaml
from sapicore.utils.sweep import Sweep
from sapicore.utils.logging import Loggable

from tree_config import Configurable, apply_config

__all__ = ("Component",)


class Component(Module, Configurable, Loggable):
    """Model component base class.

    Defines and initializes instance attributes shared by neurons and synapses.

    Parameters
    ----------
    identifier: str, optional
        Human-readable identifier for the component instance.

    configuration: dict, optional
        Configuration dictionary used during parameter initialization.

    device: str
        Specifies a hardware device on which to process this object's tensors, e.g. "cuda:0".
        Must be a valid torch.device() string value. Defaults to "cpu".

    kwargs:
        Additional instance attributes that the user may set.

    """

    def __init__(self, identifier: str = None, configuration: dict = None, device: str = "cpu", **kwargs):
        """Initializes generic instance attributes shared by all analog and spiking neuron derived classes."""
        super().__init__()

        # model-related common instance attributes.
        self.identifier = identifier
        self.configuration = configuration

        # hardware device on which to process this component.
        self.device = device

        # simulation-related tracking variables.
        self.simulation_step = 0
        self.dt = DT

        # developer may override or define arbitrary attributes at instantiation.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # initialize configurable and loggable attributes with dummy tensors (will vary across component models).
        for prop in self._config_props_:
            setattr(self, prop, torch.zeros(1, dtype=torch.float, device=self.device))

        for prop in self._loggable_props_:
            self.register_buffer(prop, torch.zeros(1, dtype=torch.float, device=self.device))

    def configure(self, configuration: dict[str, Any], log_destination: str = None):
        """Applies a configuration by adding the keys of `configuration` as instance attributes,
        initializing their values, and updating the `_config_props_` tuple to reflect the new keys.
        Also updates `_loggable_props_` for this object if the key "loggable" is provided in the dictionary.

        Parameters
        ----------
        configuration: dict or str
            Parameters to be applied to this neuron, given as a dictionary.

        log_destination: str, optional
            Path to a destination YAML file where the configuration will be saved.
            If not provided, the configuration is not saved to file.

        Note
        ----
        This method is generic and can be used without alterations by all derivative classes.

        The sub-dictionary `model` may include the special `sweep` key, which specifies settings for the
        heterogeneous initialization of certain parameters (see :class:`~utils.sweep.Sweep` for details).

        """
        # apply initial configuration if given.
        apply_config(self, configuration)

        # override loggable properties if provided in the configuration dictionary.
        if configuration.get("loggable"):
            self._loggable_props_ = configuration.get("loggable")

        # if file argument provided, back up configuration dictionary to the file `log_destination`.
        dump_yaml(self.configuration, log_destination)

    def forward(self, data: Tensor) -> dict:
        """Passes input through the component.

        Parameters
        ----------
        data: Tensor
            Input to be added to this component's numeric state tensor (e.g., `voltage`).

        Returns
        -------
        dict
            A dictionary with loggable attributes for potential use by a :class:`~pipeline.simulation.Simulator`
            handling runtime operations.

        Raises
        ------
        NotImplementedError
            The forward method must be implemented by each derived class.

        """
        raise NotImplementedError

    def heterogenize(self, unravel: bool = True):
        """Initializes component's attribute tensors heterogeneously based on a given sweep search dictionary.

        Parameters
        ----------
        unravel: bool
            When the property is a 2D tensor, dictates whether combination values should be assigned
            to individual elements (True) or overwrite entire rows (False).

            The former behavior is appropriate for synapse attributes. It is achieved using `numpy.unravel_index`,
            which maps flattened-like indices to their nD equivalents given the shape of the array/tensor.

            The latter behavior is required when values are themselves 1D vectors
            (e.g., oscillator component frequencies).

        """
        # safe to call if object has no sweep specification or empty configuration; method will return silently.
        if self.configuration and self.configuration.get("model", {}).get("sweep"):
            # the sweep object maintains a list of valid parameter combinations, each of which is a dictionary.
            sweep = Sweep(
                search_space=self.configuration.get("model", {}).get("sweep"), num_combinations=self.num_units
            )

            # modify this instance in-place.
            sweep.heterogenize(obj=self, unravel=unravel)

    def state(self) -> dict:
        """Returns loggable properties and their states.

        Returns
        -------
        dict
            Dictionary containing loggable property names and their values.

        """
        state = {}
        for prop in self._loggable_props_:
            state[prop] = getattr(self, prop)

        # return current state(s) of loggable attributes as a dictionary.
        return state
