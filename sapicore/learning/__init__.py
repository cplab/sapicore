"""Learning
===========

A learning rule is used with a :class:`~sapicore.model.SapicoreModel` to apply
learning to the neurons and synapses in the networks of the model.
"""
import torch.nn as nn

from tree_config import Configurable
from sapicore.logging import Loggable

__all__ = ('SapicoreLearning', )


class SapicoreLearning(Loggable, Configurable, nn.Module):
    """A Sapicore learning instance.
    """

    def apply_learning(self, **kwargs):
        """Applies the learning rule to the model.

        Should be overwritten to apply the rule.
        """
        pass

    def initialize_state(self, **kwargs):
        """Initializes the learning rule for usage.
        """
        pass
