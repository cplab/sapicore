"""Synapses connect neurons or ensembles to each other."""
from collections import deque
from typing import Callable

import torch
import numpy as np

from torch import tensor, Tensor
from torch.nn.init import xavier_uniform_

from sapicore.engine.component import Component
from sapicore.engine.neuron import Neuron

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

    weights: Tensor
        2D weight matrix.

    connections: Tensor.
        2D connectivity mask matrix.

    weight_max: float or Tensor
        Positive weight value limit for any synapse element (applies throughout the simulation).

    weight_min: float or Tensor
        Negative weight value limit for any synapse element (applies throughout the simulation).

    delay_ms: float or Tensor
        Generic transmission delay, managed by the synapse object using queues.
        Transmission delays are useful for some temporal coding schemes.

    simple_delays: bool
        Whether transmission delay values from one presynaptic element are the same for all postsynaptic targets.
        Provides a speed-control tradeoff (one matrix multiplication vs. N vector multiplications).

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

    _config_props_: tuple[str] = ("weight_max", "weight_min", "delay_ms", "simple_delays")
    _loggable_props_: tuple[str] = ("weights", "connections", "output")

    # declare loggable instance attributes registered as torch buffers.
    connections: Tensor

    def __init__(
        self,
        src_ensemble: [Neuron] = None,
        dst_ensemble: [Neuron] = None,
        weights: [Tensor] = None,
        connections: [Tensor] = None,
        weight_max: float = 1000.0,
        weight_min: float = -1000.0,
        delay_ms: int = 0,
        simple_delays: bool = True,
        weight_init_method: Callable = xavier_uniform_,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # model-related common instance attributes.
        self.learning = False

        self.src_ensemble = src_ensemble
        self.dst_ensemble = dst_ensemble

        try:
            self.matrix_shape = (dst_ensemble.num_units, src_ensemble.num_units)

        except AttributeError:
            self.matrix_shape = (1, 1)

        self.num_units = torch.prod(torch.tensor(self.matrix_shape), 0).item()

        self.weight_max = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + weight_max
        self.weight_min = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + weight_min
        self.weight_init_method = weight_init_method  # xavier_uniform by default.

        # [FAST] synaptic delay ring buffer initialization.
        self.delay_ms = delay_ms
        self.delay_buffer = None
        self.delay_indices = None

        # [LEGACY] whether transmission delays from one presynaptic element are shared among postsynaptic targets.
        self.simple_delays = simple_delays
        self.delay_queue = None
        self.delayed_data = None
        self.legacy = kwargs.pop("legacy", False)

        self._compute_delays()

        # container for data pulled from the queue in this simulation step (input as seen by `dst_ensemble`).
        self.delayed_data = torch.zeros(self.matrix_shape[1], dtype=torch.float, device=self.device)

        # float matrix containing synaptic weights.
        if weights is not None:
            if not isinstance(weights, Tensor):
                w_init = torch.ones(self.matrix_shape, device=self.device) * weights
            else:
                w_init = weights.to(self.device)
            self.register_buffer("weights", w_init)
        else:
            self.register_buffer(
                "weights",
                self.weight_init_method(tensor=torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device)),
            )

        # binary connectivity matrix marking enabled/disabled connections (integer mask, all-to-all by default).
        # initialize connections from constructor input, if given.
        if connections is not None:
            self.register_buffer("connections", connections.to(self.device))
        else:
            self.register_buffer("connections", torch.ones_like(self.weights, dtype=torch.bool))

        # if given, apply connection strategy; all connections enabled by default.
        conn_mode = kwargs.pop("conn_mode", "all")
        if conn_mode:
            self.connect(conn_mode, kwargs.pop("conn_prop", 1))

        # output 1D tensor containing data for the destination ensemble.
        self.register_buffer("output", torch.zeros(self.matrix_shape[0], dtype=torch.float, device=self.device))

        # clamp weights to acceptable range given this synapse's min and max.
        self.weights = torch.clamp(self.weights, self.weight_min, self.weight_max)

        # if synapse represents a connection from an ensemble to itself, zero out diagonal of mask matrix.
        if self.src_ensemble is self.dst_ensemble:
            self.connections.fill_diagonal_(0)

        # if autograd parameter provided to constructor at build time, turn autograd on or off for the weights.
        if hasattr(self, "autograd"):
            self.weights.requires_grad_(self.autograd)

        # specifies which loggable attribute should be considered this unit's output.
        self.output_field = "output"

    def _compute_delays(self):
        """Compute synaptic delay buffers. Backward compatible with Sapicore < 0.4.0.

        Note
        ----
        Invoked in :meth:`heterogenize`, as it can potentially alter delay values after synapse instantiation.

        """
        # initialize default transmission delay values as a tensor of size `src_ensemble.num_units`.
        self.delay_ms = torch.zeros(self.matrix_shape[1], dtype=torch.float, device=self.device) + self.delay_ms

        if self.legacy:
            if self.simple_delays:
                # maintains delayed data queues.
                self.delay_queue = [
                    deque(torch.zeros(delay, device=self.device)) for delay in (self.delay_ms // self.dt).int()
                ]
            else:
                # the extensive delay setting, where delayed input views may vary across postsynaptic elements.
                self.delay_ms = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + self.delay_ms
                self.delayed_data = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device)
                self.delay_queue = [
                    deque(torch.zeros(delay.int(), device=self.device))
                    for delay in (self.delay_ms / self.dt).flatten().int()
                ]
        else:
            delay_ms_int = self.delay_ms.int()
            self.max_delay = delay_ms_int.max().item() + 1

            self.delay_indices = self.delay_ms.clone().to(self.device).long()
            self.delay_buffer = torch.zeros(
                (self.matrix_shape[1], self.max_delay), dtype=torch.float, device=self.device
            )

    def connect(self, mode: str = "all", prop: float = 1.0):
        """Applies a predefined connectivity mask strategy.

        Parameters
        ----------
        mode: str
            Sapicore provides four built-in connectivity mask initialization options: "all", "one", "prop", "rand".
            "all" enables all connections (matrix of 1s) except self-connections (diagonal) if the destination and
            source ensemble are the same object. "one" connects the i-th source neuron to the i-th destination neuron,
            zeroing out all matrix elements **except** the diagonal. "prop" connects each source neuron to
            `prop`*`dst_ensemble.num_units` randomly selected destination neurons. "rand" sets a proportion `prop`
            of the matrix elements to 1 with no balance constraints (unlike "prop").

        prop:
            Desired proportion of neurons in destination receiving a connection from a particular neuron
            in the source ensemble. Rounded to an integer using np.round.

            E.g., if prop = 0.2 and matrix_shape is (100, 5), 20 random row elements (destination neurons)
            in the connectivity mask will be set to 1.

        Raises
        ------
        ValueError
            If `mode` was set to a value other than "all", "one", "prop", or "rand".

        Warning
        -------
        Attempting to initialize one-to-one connectivity between ensembles of different sizes will necessarily
        result in some unconnected neurons. The faux-diagonal will be set to 1s but there will be all 0s rows/cols,
        depending on whether the source or destination ensemble is bigger.

        """
        match mode:
            case "all":
                self.connections = torch.ones_like(self.connections, device=self.device, dtype=torch.bool)

            case "one":
                num_src = self.src_ensemble.num_units
                num_dst = self.dst_ensemble.num_units

                # produces identity-like non-square matrices if applicable.
                self.connections = torch.eye(num_src, device=self.device, dtype=torch.bool).repeat_interleave(
                    num_dst // num_src, dim=0
                )

            case "prop":
                selection_size = int(np.round(prop * self.matrix_shape[0]))
                torch.randperm(self.matrix_shape[0])

                self.connections = torch.zeros_like(self.connections, device=self.device, dtype=torch.bool)
                for i in range(self.matrix_shape[1]):
                    self.connections[torch.randperm(self.matrix_shape[0])[:selection_size], i] = 1

            case "rand":
                num_enabled = int(np.round(prop * self.weights.numel()))
                ids_enabled = np.random.choice(np.arange(self.weights.numel()), size=num_enabled, replace=False)

                self.connections = torch.zeros_like(self.connections, device=self.device, dtype=torch.bool)
                self.connections[np.unravel_index(ids_enabled, shape=self.matrix_shape)] = 1.0

            case _:
                raise ValueError(f"Incorrect mode specified {mode}. Use 'all', 'one', 'prop', or 'rand'.")

    def heterogenize(self, num_combinations: int, unravel: bool = True):
        """Diversifies parameters in accordance with this synapse configuration dictionary and recomputes
        the spike delay queue in case values were altered."""
        super().heterogenize(num_combinations=self.num_units, unravel=unravel)

        # recompute delay steps and reinitialize ring buffer (or queues in legacy).
        self._compute_delays()

    def set_learning(self, state: bool = False) -> None:
        """Switch weight updates on or off, e.g. before a training/testing round commences.

        Parameters
        ----------
        state: bool
            Toggle learning on if True, off if False.

        Note
        ----
        Fine-grained control over which synapse elements are toggled on will be added in the future, to support
        more sophisticated algorithms. Currently, the global learning switch is meant to be used, e.g.,
        when feeding test buffer to a trained network with STDP synapses.

        """
        self.learning = state

    def set_weights(self, weight_initializer: Callable, *args, **kwargs) -> None:
        """Initializes 2D weight matrix for this synapse instance.

        Parameters
        ----------
        weight_initializer: Callable
            Any weight initialization method from :mod:`torch.nn.init`. Args and kwargs are passed along to it.

        """
        self.weights = weight_initializer(tensor=self.weights, *args, **kwargs)

    def queue_input(self, current_data: Tensor) -> Tensor:
        """Loads incoming spikes into the back of a ring buffer at the correct index,
        and returns the delayed spikes ready for transmission (from the front of the queue).

        Note
        ----
        Legacy variant uses a list of queues to implement transmission delays and "release" input appropriately.

        Initializes a list whose Nth element is a queue pre-filled with ``self.delay_ms[N] // DT`` zeros.
        On each forward iteration, call this method. It will enqueue the current source input data value and
        :meth:`popleft` the head of the queue.

        Returns
        -------
        Tensor
            1D tensor of input that occurred an appropriate number of steps ago for each unit.

        """
        if not self.legacy:
            # Add the incoming spikes at the current positions indicated by delay_indices
            n_synapses = self.matrix_shape[1]
            self.delay_buffer[torch.arange(n_synapses), self.delay_indices % self.max_delay] = current_data

            # Transmit the spikes at index 0 (front of the queue)
            spikes_to_transmit = self.delay_buffer[:, 0]

            # Rotate the delay buffer to the left (advance the spikes)
            self.delay_buffer = torch.cat(
                [self.delay_buffer[:, 1:], torch.zeros((n_synapses, 1), device=self.device)], dim=1
            )
            return spikes_to_transmit

        else:
            if self.simple_delays:
                # append updated presynaptic data to the queues, each corresponding to one presynaptic element.
                for i, value in enumerate(current_data):
                    if i >= len(self.delay_queue):
                        breakpoint()
                    self.delay_queue[i].append(value)

            else:
                # append updated presynaptic data to the delayed view of each postsynaptic element (src X dst queues).
                for i in range(len(self.delay_queue)):
                    multi_index = np.unravel_index(i, self.matrix_shape)
                    self.delay_queue[i].append(current_data[multi_index[1]])

            # return appropriate data tensor while popping the head of the line.
            valid_data = [item.popleft() for item in self.delay_queue]
            return tensor(valid_data, device=self.device)

    # child classes would typically only implement one or more of the methods below this line.
    def update_weights(self) -> Tensor:
        """Static weight update rule. Placeholder for plastic synapse derivative classes (e.g., STDP)."""
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
        The simple transmission delay implementation, for runtime efficiency, assumes that recipient synapse elements
        all have a shared "view" of incoming activity in the presynaptic ensemble. This means a single matrix
        multiplication (dst x src) by (src x 1) = (dst x 1) is performed. A smaller queue list of length
        `src_ensemble.num_units` is maintained, as opposed to `src_ensemble.matrix_shape`.

        If `simple_delays` is toggled **off**, each neuron in the recipient ensemble may have a different delayed
        view of the source ensemble activity. This means that N vector multiplications must be performed:
        sig(W[k,i]*dd[i]) where N is the number of elements in the source ensemble, k the index of a destination
        element, i the index of a source element, W a weight, and dd the delayed data transmitted from the
        presynaptic neuron i to the postsynaptic neuron k.

        """
        # use a list of queues to track transmission delays and "release" input after the fact.
        self.delayed_data = self.queue_input(data)

        # enforce weight limits.
        self.weights = torch.clamp(self.weights, self.weight_min, self.weight_max)

        if self.simple_delays:
            # mask non-connections (repeated on every iteration in case mask was updated during simulation).
            self.output = torch.matmul(self.weights * self.connections, self.delayed_data)

        else:
            self.delayed_data = self.delayed_data.reshape(self.matrix_shape)

            # this loop is unavoidable, as N vector multiplications with different delayed_data are needed.
            for i in range(self.matrix_shape[0]):
                self.output[i] = torch.matmul(self.weights[i, :] * self.connections[i, :], self.delayed_data[i, :])

        # advance simulation step and return output dictionary.
        self.simulation_step += 1

        return self.loggable_state()
