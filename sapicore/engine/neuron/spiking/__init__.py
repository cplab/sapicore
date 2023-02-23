""" Spiking neurons emit binary spikes once their numeric state surpasses a threshold.

Spiking neurons inherit all properties of their parent class :class:`~neuron.Neuron`. In addition to `voltage`,
they also own the registered buffer `spiked`.

"""
import torch
from torch import Tensor

from sapicore.engine.neuron import Neuron

__all__ = ("SpikingNeuron",)


class SpikingNeuron(Neuron):
    """Generic spiking neuron.

    Defines instance attributes shared by all derived spiking neuron classes.

    Parameters
    ----------
    spiked: Tensor
        A binary representation of spike events, registered as a PyTorch buffer.

    """

    _loggable_props_: tuple[str] = ("input", "voltage", "spiked")

    # declare instance attributes to be registered as pytorch buffers.
    spiked: Tensor

    def __init__(self, **kwargs):
        """Invokes the parent constructor to initialize common instance attributes."""
        # register universal loggables and configurables using the parent method(s).
        super().__init__(**kwargs)

        # state-related attributes unique to spiking neurons, over and above base components and neurons.
        self.register_buffer("spiked", torch.zeros(1, dtype=torch.int8, device=self.device))

    def forward(self, data: Tensor) -> dict:
        """Passes external input through the neuron unit.

        Returns
        -------
        dict
            A dictionary with loggable attributes for potential use by the :class:`~simulation.Simulator` object
            handling runtime operations (e.g., selectively updating structural connectivity during neurogenesis).

        Raises
        ------
        NotImplementedError
            The forward method must be implemented by each derived class.
        """
        raise NotImplementedError
