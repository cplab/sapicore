"""Component objects handle universal attributes and logic common to classes that may serve as model components.
Those include neurons, synapses, and their derivatives."""
from typing import Any
import os

import torch
from torch import Tensor
from torch.nn import Module

from sapicore.utils.constants import DT
from sapicore.utils.io import save_yaml
from sapicore.utils.sweep import Sweep
from sapicore.utils.loggable import Loggable

from tree_config import Configurable, apply_config

__all__ = ("Component",)


class Component(Module, Configurable, Loggable):
    """Model component base class.

    Defines instance attributes and methods shared by all model components.
    Its primary purpose is to implement generic operations on loggable and configurable attributes.

    Parameters
    ----------
    identifier: str, optional
        Human-readable identifier for the component instance.

    configuration: dict, optional
        Configuration dictionary used during parameter initialization.

    device: str, optional
        Specifies a hardware device on which to process this object's tensors, e.g. "cuda:0".
        Must be a valid torch.device() string value. Defaults to "cpu".

    kwargs:
        Additional instance attributes that the user may set.

    See Also
    --------
    :class:`~tree_config.Configurable`
    :class:`~utils.loggable.Loggable`
    :class:`~utils.sweep.Sweep`

    """

    def __init__(self, identifier: str = None, configuration: dict = None, device: str = "cpu", **kwargs):
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

        # initialize configurable and loggable attributes with dummy tensors.
        for prop in self._config_props_:
            setattr(self, prop, torch.zeros(1, dtype=torch.float, device=self.device))

        for prop in self._loggable_props_:
            self.register_buffer(prop, torch.zeros(1, dtype=torch.float, device=self.device))

    def configure(self, configuration: dict[str, Any], log_destination: str = None):
        """Applies a configuration to this object by adding the keys of `configuration` as instance attributes,
        initializing their values, and updating the `_config_props_` tuple to reflect the new keys.

        Also updates `_loggable_props_` for this object if the configuration includes the key "loggable".

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
        heterogeneous initialization of certain parameters in the form of a dictionary.

        """
        # apply this object's configuration by adding or updating instance variables.
        apply_config(self, configuration)

        # override _loggable_properties_ if provided in the configuration dictionary.
        if configuration.get("loggable"):
            self._loggable_props_ = configuration.get("loggable")

        # if `log_destination` was passed, save the configuration dictionary to that location.
        if os.path.exists(log_destination):
            save_yaml(self.configuration, log_destination)

    def forward(self, data: Tensor) -> dict:
        """Processes an input, updates the state of this component, and advances the simulation by one step.

        Parameters
        ----------
        data: Tensor
            Input to be processed (e.g., added to a neuron's numeric state tensor `voltage`).

        Returns
        -------
        dict
            A dictionary whose keys are loggable attributes and whose values are their states as of this time step.
            For potential use by a :class:`~pipeline.simulation.Simulator` or any other :class:`~pipeline.Pipeline`
            script handling runtime operations.

        Raises
        ------
        NotImplementedError
            The forward method must be implemented by each derived class.

        """
        raise NotImplementedError

    def heterogenize(self, unravel: bool = True):
        """Edits configurable tensor attributes based on a sweep search dictionary if one was provided by the user
        within this object's `configuration` dictionary.

        Note
        ----
        If a `sweep` key is not present in `configuration`, this method will pass silently, retaining the existing
        configurable values. It can be invoked safely from generic methods (e.g.,
        :meth:`~engine.network.Network.build`).

        Parameters
        ----------
        unravel: bool
            When the property is a 2D tensor, dictates whether combination values should be assigned
            to individual elements (True) or overwrite entire rows (False).

            The former behavior is appropriate for synapse attributes. It is achieved using `numpy.unravel_index`,
            which maps flattened-like indices to their nD equivalents given the shape of the array.

            The latter behavior is called for when values are themselves 1D vectors (e.g., oscillator frequencies).

        Warning
        -------
        This method mutates the underlying component object. To simply view the combinations, initialize a
        :class:`~utils.sweep.Sweep` object with your desired search space dictionary and number of combinations.

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
        """Returns a dictionary of this object's loggable properties and their states as of this simulation step.

        Returns
        -------
        dict
            Dictionary containing loggable property names and their values.

        """
        return {prop: getattr(self, prop) for prop in self._loggable_props_}
