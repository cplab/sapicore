"""Neuron
=========

A neuron is a basic neuronal unit that can be connected to each other with a
:class:`~sapicore.synapse.SapicoreSynapse` and is a node in a
:class:`~sapicore.network.SapicoreNetwork`.

"""
import torch.nn as nn
import torch

from tree_config import Configurable
from sapicore.logging import Loggable

__all__ = ('SapicoreNeuron', )


class SapicoreNeuron(Loggable, Configurable, nn.Module):
    """Neuron baseclass.
    """

    def forward(self, data: torch.tensor) -> torch.tensor:
        """The forward method, similar to pytorch models, that passes the input
        through the neuron.
        """
        raise NotImplementedError

    def initialize_state(self, **kwargs) -> None:
        """Initializes the neuron before it is used.
        """
        pass
