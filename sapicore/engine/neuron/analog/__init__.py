""" Analog neurons emit real numbers, usually onto static synapses.

Analog neurons inherit all properties of their parent class :class:`~engine.neuron.Neuron`, extending it by
implementing a generic forward method that simply adds incoming input to their numeric state tensor `voltage`.

Analog neurons may perform normalization or provide otherwise transformed input to downstream layers.

"""
from torch import Tensor
from sapicore.engine.neuron import Neuron

__all__ = ("AnalogNeuron",)


class AnalogNeuron(Neuron):
    """Generic analog neuron endowed with a trivial :meth:`forward` method.

    Defines instance attributes shared by all derived analog neuron classes.

    Parameters
    ----------
    spiked: Tensor
        A binary representation of spike events, registered as a PyTorch buffer.

    """

    _loggable_props_: tuple[str] = ("input", "voltage")

    def __init__(self, **kwargs):
        """Invokes the parent :class:`~engine.neuron.Neuron` constructor to initialize common instance attributes."""
        super().__init__(**kwargs)

    def forward(self, data: Tensor) -> dict:
        """Updates the numeric state stored in the instance attribute tensor `voltage` to `data`.

        These default analog neurons integrate the total input impinging on them on every simulation step.


        Parameters
        ----------
        data: Tensor
            External input current to be added to this unit's numeric state tensor `voltage`.

        Raises
        ------
        RuntimeError
            If `data` tensor is not on the same hardware device as `voltage` tensor.

        Returns
        -------
        dict
            Dictionary containing the numeric state tensor `voltage`.

        """
        # update internal representation of input current for tensorboard logging purposes.
        self.input = data
        self.voltage = data

        self.simulation_step += 1

        # return current state(s) of loggable attributes as a dictionary.
        return self.loggable_state()
