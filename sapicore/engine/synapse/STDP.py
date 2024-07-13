""" Synapses with spike-timing-dependent plasticity (STDP). """
import torch
from torch import Tensor

from sapicore.engine.synapse import Synapse

__all__ = ("STDPSynapse",)


class STDPSynapse(Synapse):
    """STDP synapse implementation.

    All STDP attributes are initialized as 2D tensors covering the source and destination ensembles, to allow for
    easily implementable heterogeneity in synaptic properties.

    STDP synapses own the following attributes, on top of those inherited from Synapse:

    Parameters
    ----------
    tau_plus: float or Tensor
        Time constant of potentiation STDP time window.

    tau_minus: float or Tensor
        Time constant of depression STDP time window.

    mu_plus: float or Tensor
        Positive dependence exponent (e.g., 1.0 for multiplicative STDP, 0.0 for additive STDP)

    mu_minus: float or Tensor
        Negative dependence exponent (e.g., 1.0 for multiplicative STDP, 0.0 for additive STDP)

    alpha_plus: float or Tensor
        Limit on magnitude of weight modification for positive spike time difference.

    alpha_minus: float or Tensor
        Limit on magnitude of weight modification for negative spike time difference.

    """

    _config_props_: tuple[str] = (
        "weight_max",
        "weight_min",
        "delay_ms",
        "tau_plus",
        "tau_minus",
        "mu_plus",
        "mu_minus",
        "alpha_plus",
        "alpha_minus",
    )
    _loggable_props_: tuple[str] = ("weights", "connections", "output")

    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        mu_plus: float = 0.0,
        mu_minus: float = 0.0,
        alpha_plus: float = 0.01,
        alpha_minus: float = 0.01,
        **kwargs
    ):
        """Constructs an STDP synapse object connecting two ensembles to each other or an ensemble to itself.

        :class:`~engine.synapse.Synapse` attributes should be given as keyword arguments by the calling method
        unless default values are acceptable.
        """
        super().__init__(**kwargs)

        # configurable attributes specific to STDP, over and above a static synapse.
        self.tau_plus = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + tau_plus
        self.tau_minus = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + tau_minus

        self.mu_plus = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + mu_plus
        self.mu_minus = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + mu_minus

        self.alpha_plus = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + alpha_plus
        self.alpha_minus = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device) + alpha_minus

        # state attributes specific to STDP synapses, over and above Synapse.
        self.src_last_spiked = torch.zeros(self.matrix_shape[1], dtype=torch.float, device=self.device)
        self.dst_last_spiked = torch.zeros(self.matrix_shape[0], dtype=torch.float, device=self.device)

        # enable or disable learning (weight updates).
        # in typical cases, should generally be on during training and off during testing.
        self.learning = True

        # keep track of stale spike pairs, so that weights are only updated on new spike events.
        # alternatively, set `is_discontinuous` false to update weights on every simulation step.
        self.is_discontinuous = kwargs.get("is_discontinuous", True)
        self.stale = torch.ones_like(self.weights).bool()

    def update_weights(self) -> Tensor:
        """STDP weight update implementation.

        Returns
        -------
        Tensor
            2D tensor containing weight differences (dW).

        """
        # update last spike time to current simulation step iff a spike has just arrived at the synapse.
        self.src_last_spiked = self.src_last_spiked + self.delayed_data * (-self.src_last_spiked + self.simulation_step)

        # update last spike time only for postsynaptic units that spiked in this model iteration.
        self.dst_last_spiked = self.dst_last_spiked + self.dst_ensemble.spiked * (
            -self.dst_last_spiked + self.simulation_step
        )

        # ensure action is only taken on pre-post pairs for whom one or both units spiked in this cycle.
        if self.is_discontinuous:
            self.stale[:] = True
            self.stale[:, torch.argwhere(torch.as_tensor(self.src_last_spiked == self.simulation_step))] = False
            self.stale[torch.argwhere(torch.as_tensor(self.dst_last_spiked == self.simulation_step)), :] = False
        else:
            self.stale[:] = False

        # initialize delta weight matrix (destination X source).
        delta_weight = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device)

        if torch.any(~self.stale) or not self.is_discontinuous:
            # subtract postsynaptic last spike time stamps from presynaptic.
            # transpose dst_last_spiked to a column vector, extend column-wise, then subtract from src_last_spiked.
            dst_tr = self.dst_last_spiked.reshape(self.dst_last_spiked.shape[0], 1)
            delta_spike = -dst_tr.repeat(dst_tr.shape[1], self.src_ensemble.num_units) + self.src_last_spiked

            # spike time differences for the potentiation and depression cases (pre < post, pre > post, respectively).
            ltp_diffs = (delta_spike < 0.0).int() * delta_spike
            ltd_diffs = (delta_spike > 0.0).int() * delta_spike

            # add to total delta weight matrix (diffs are in simulation steps, tau are in ms).
            delta_weight = delta_weight + ~self.stale * (delta_spike < 0.0).int() * (
                self.alpha_plus * torch.exp(ltp_diffs / (self.tau_plus / self.dt))
            )
            delta_weight = delta_weight + ~self.stale * (delta_spike > 0.0).int() * (
                -self.alpha_minus * torch.exp(-ltd_diffs / (self.tau_minus / self.dt))
            )
            self.weights = self.weights.add(delta_weight)

        return delta_weight

    def forward(self, data: Tensor) -> dict:
        """Updates weights if need be, then calls the parent :class:`~engine.synapse.Synapse` :meth:`forward` method.

        Parameters
        ----------
        data: Tensor
            Source ensemble spike output to be processed by this STDP synapse.

        Returns
        -------
        dict
            Dictionary containing `weights`, `connections`, and `output` for further processing.

        """
        if self.learning:
            self.update_weights()

        return super().forward(data)
