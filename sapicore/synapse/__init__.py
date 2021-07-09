"""Synapse
==========

A synapse connects two :class:`~sapicore.neuron.SapicoreNeuron`
and is the link in a neuronal :class:`~sapicore.network.SapicoreNetwork`.
"""

import torch.nn as nn
import torch

from tree_config import Configurable
from sapicore.logging import Loggable

__all__ = ('SapicoreSynapse', )


class SapicoreSynapse(Loggable, Configurable, nn.Module):
    """Synapse baseclass.
    """

    def initialize_state(self, **kwargs) -> None:
        """Initializes the synapse before it is used.
        """
        pass

    def forward(self, data: torch.tensor) -> torch.tensor:
        """The forward method, similar to pytorch models, that process the
        data through the synapse.
        """
        raise NotImplementedError
