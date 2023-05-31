""" Spiking neurons emit all-or-none events once their numeric state surpasses a critical threshold.

Spiking neurons inherit all properties of their parent class :class:`~engine.neuron.Neuron`. In addition to `voltage`,
they also own the binary integer tensor `spiked`, which maintains their emitted action potentials.

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
        # register universal loggables and configurables using the parent method(s).
        super().__init__(**kwargs)

        # state-related attributes unique to spiking neurons, over and above base components and neurons.
        self.register_buffer("spiked", torch.zeros(1, dtype=torch.int8, device=self.device))

        # update preferred output field to spiked.
        self.output_field = "spiked"

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
            For potential use by a :class:`~pipeline.simulation.SimpleSimulator` or any other
            :class:`~pipeline.Pipeline` script handling runtime operations.

        Raises
        ------
        NotImplementedError
            The forward method must be implemented by each derived class.

        """
        raise NotImplementedError
