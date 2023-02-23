"""Synapses connect neurons or ensembles to each other."""
import logging

from collections import deque
from typing import Callable

import torch
import numpy as np

from torch import tensor, Tensor
from torch.nn.init import xavier_uniform_

from sapicore.engine.component import Component
from sapicore.engine.neuron import Neuron

from sapicore.utils.constants import DT

__all__ = ("Synapse",)


class Synapse(Component):
    """Static synapse base class.

    This class provides basic parameters and logic for connecting ensemble instances. Static synapses keep their
    initial weights throughout the simulation unless manually adjusted. Synapse instances have 2D weight and
    connectivity matrices, reflecting their role as layer-to-layer connectors.

    Synapses own the following parameters, over and above those inherited from :class:`~engine.component.Component`:

    Parameters
    ----------
    src_ensemble: Neuron
        Reference to the presynaptic (sending) ensemble.

    dst_ensemble: Neuron
        Reference to postsynaptic ensemble (receiving) ensemble.

    weight_max: float or Tensor
        Positive weight value limit for any synapse element (applies throughout the simulation).

    weight_min: float or Tensor
        Negative weight value limit for any synapse element (applies throughout the simulation).

    delay_ms: float or Tensor
        Generic transmission delay, managed by the synapse object using queues.

    weight_init_method: Callable
        Weight initialization method from `torch.nn.init`.

    connections: Tensor
        2D binary integer mask matrix indicating which source unit -> destination unit connections are enabled.
        Loggable attribute.

    weights: Tensor
        2D float matrix storing a weight for each pair of source and destination ensemble units.
        Loggable attribute.

    output: Tensor
        1D tensor storing the result of W*x matrix multiplication in the current simulation step.
        Loggable attribute.

    """

    _config_props_: tuple[str] = ("weight_max", "weight_min", "delay_ms")
    _loggable_props_: tuple[str] = ("weights", "connections", "output")

    # declare loggable instance attributes registered as torch buffers.
    connections: tensor
    weights: tensor
    output: tensor

    def __init__(
        self,
        src_ensemble: [Neuron] = None,
        dst_ensemble: [Neuron] = None,
        weight_max: float = 1.0,
        weight_min: float = -1.0,
        delay_ms: float = 2.0,
        weight_init_method: Callable = xavier_uniform_,
        **kwargs
    ):
        super().__init__(**kwargs)

        # model-related common instance attributes.
        self.src_ensemble = src_ensemble
        self.dst_ensemble = dst_ensemble

        try:
            self.matrix_shape = (dst_ensemble.num_units, src_ensemble.num_units)

        except AttributeError:
            logging.info("Attempted to initialize synapse object with no source and destination ensembles.")
            self.matrix_shape = (1, 1)

        self.num_units = torch.prod(torch.tensor(self.matrix_shape), 0).item()

        self.weight_max = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + weight_max
        self.weight_min = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + weight_min
        self.weight_init_method = weight_init_method  # xavier_uniform by default.

        # simulation-related common instance attributes.
        # transmission delays are necessary for realism and for some temporal coding schemes.
        self.delay_ms = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + delay_ms

        # container for data pulled from the queue in this simulation step.
        self.delayed_data = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device)

        # maintains delayed data queue with spikes or analog output.
        self.delay_queue = [
            deque(torch.zeros(delay.int(), device=self.device)) for delay in (self.delay_ms / DT).flatten().int()
        ]

        # binary connectivity matrix marking enabled/disabled connections (integer mask, all-to-all by default).
        self.register_buffer("connections", torch.ones(self.matrix_shape, dtype=torch.uint8, device=self.device))

        # if synapse represents a connection from an ensemble to itself, zero out diagonal of mask matrix.
        if self.src_ensemble is self.dst_ensemble:
            self.connections.fill_diagonal_(0)

        # float matrix containing synaptic weights.
        self.register_buffer("weights", torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device))

        # default initialization method for weights (can be overriden).
        self.weights = self.weight_init_method(tensor=self.weights)

        # output 1D tensor containing data for the destination ensemble.
        self.register_buffer("output", torch.zeros(self.matrix_shape[0], dtype=torch.float, device=self.device))

        # developer may override or define arbitrary attributes at instantiation.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # if autograd parameter provided to constructor at build time, turn autograd on or off for the weights.
        if hasattr(self, "autograd"):
            self.weights.requires_grad_(self.autograd)

    def heterogenize(self, unravel: bool = True):
        """Diversifies parameters in accordance with this synapse configuration dictionary and recomputes
        the spike delay queue in case values were altered."""
        super().heterogenize(unravel=unravel)

        # recompute delay steps and reinitialize queue.
        self.delay_queue = [
            deque(torch.zeros(delay.int(), device=self.device)) for delay in (self.delay_ms / DT).flatten().int()
        ]

    def initialize(self) -> None:
        """Static synapse state initialization, callable with or without a configuration dictionary.

        Applies all extra keyword arguments to object. Default behavior: initializes delays ~U(1ms, 20ms) and a
        2D weight matrix using :func:`torch.nn.init.xavier_uniform_(gain=1.0)`. Retains default all-to-all (1s)
        2D connectivity mask unless lateral synapse, in which case zeros out diagonal to disable unit self-connections.

        """
        # if synapse represents a connection from an ensemble to itself, zero out diagonal of mask matrix.
        if self.src_ensemble is self.dst_ensemble:
            self.connections.fill_diagonal_(0)

        # if autograd parameter provided to constructor at build time, turn autograd on or off for the weights.
        if hasattr(self, "autograd"):
            self.weights.requires_grad_(self.autograd)

        # recompute delay steps and reinitialize queue in case user overwrote the `delay_ms` setting.
        self.delay_queue = [
            deque(torch.zeros(delay.int(), device=self.device)) for delay in (self.delay_ms / DT).flatten().int()
        ]

    def set_weights(self, weight_initializer: Callable, *args, **kwargs) -> None:
        """Initializes 2D weight matrix for this synapse instance.

        Parameters
        ----------
        weight_initializer: Callable
            Any weight initialization method from :mod:`torch.nn.init`. Args and kwargs are passed along to it.

        """
        self.weights = weight_initializer(tensor=self.weights, *args, **kwargs)

    def queue_input(self, current_data: Tensor) -> Tensor:
        """Uses a list of queues to implement transmission delays and "release" input appropriately.

        Initializes a list whose Nth element is a queue pre-filled with ``self.delay_ms[N] // DT`` zeros.
        On each forward iteration, call this method. It will enqueue the current source input data value and
        :meth:`popleft` the head of the queue.

        Returns
        -------
        Tensor
            1D tensor of input that occurred an appropriate number of steps ago for each unit.

        """
        # FIX since this comprehensive delay method is x3 slower, need to reintroduce old one and allow user to pick.
        # add updated presynaptic data to the delayed view of each postsynaptic element.
        for i in range(len(self.delay_queue)):
            multi_index = np.unravel_index(i, self.matrix_shape)
            self.delay_queue[i].append(current_data[multi_index[1]])

        # return appropriate data tensor while popping the head of the line.
        valid_data = [item.popleft() for item in self.delay_queue]
        return tensor(valid_data, device=self.device)

    # child classes would typically only implement one or more of the methods below this line.
    def update_weights(self) -> Tensor:
        """Static weight update rule, placeholder for child classes (e.g., STDP)."""
        pass

    def forward(self, data: Tensor) -> dict:
        """Static synapse forward method. Amounts to a simple matrix multiplication.

        When called with synapses that have nonzero transmission delays, variably-old data will be used as
        dictated by the queue maintained by this instance.

        Parameters
        ----------
        data: Tensor
            Output vector from the input layer in the previous iteration (`spiked` buffer, containing 0s and 1s).

        Returns
        -------
        dict
            Dictionary containing `weights`, `connections`, and `output` for further processing.

        Note
        ----
        Each neuron in the recipient ensemble may have a different delayed view of the source ensemble activity.
        This means that N vector multiplications must be performed: sig(W[k,i]*dd[i]) where N is the number of
        elements in the source ensemble, k the index of a destination element, i the index of a source element,
        W a weight, and dd the delayed data transmitted from presynaptic neuron i to postsynaptic neuron k.

        Warning
        -------
        The previous implementation assumed, for simplicity and runtime efficiency, that recipient synapse elements
        all have a shared "view" of incoming activity--that meant matmul (dst x src) by (src x 1) = (dst x 1),
        also maintaining a smaller queue of size `src_ensemble.num_units` as opposed to `src_ensemble.matrix_shape`
        in this version.

        """
        # use a list of queues to track transmission delays and "release" input after the fact.
        self.delayed_data = self.queue_input(data).reshape(self.matrix_shape)

        # mask non-connections (repeated on every iteration in case connection mask updated during simulation).
        self.weights = self.weights.multiply(self.connections)

        # this loop is unavoidable, as N vector multiplications with different delayed_data are needed (see docstring).
        for i in range(self.matrix_shape[0]):
            self.output[i] = torch.matmul(self.weights[i, :], self.delayed_data[i, :])

        # enforce weight limits by adding the difference from threshold for cells exceeded.
        weight_plus = self.weights > self.weight_max
        weight_minus = self.weights < self.weight_min

        self.weights = self.weights.add(weight_plus.int() * (self.weight_max - self.weights))
        self.weights = self.weights.add(weight_minus.int() * (self.weight_min - self.weights))

        # advance simulation step and return output dictionary.
        self.simulation_step += 1

        return self.state()
