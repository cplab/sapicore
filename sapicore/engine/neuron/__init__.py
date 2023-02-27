"""Neurons are units that receive, integrate, and emit signals.

Neuron instance attributes are 1D tensors of arbitrary length, initialized with a single element by default.

Neurons maintain a numeric state akin to a membrane potential in the float32 tensor `voltage`,
and sometimes a binary integer tensor `spiked` representing emitted action potential events.

All classes derived from :class:`~engine.neuron.Neuron` must implement the parent
:meth:`~engine.component.Component.forward` method.

"""
from typing import Callable
from torch import Tensor

from sapicore.engine.component import Component
from sapicore.utils.integration import Integrator, RungeKutta

__all__ = ("Neuron",)


class Neuron(Component):
    """Generic neuron base class.

    Defines and initializes instance attributes shared by all neuron objects, over and above the universal
    specifications handled by the parent class :class:`~engine.component.Component`.

    Parameters
    ----------
    equation: Callable or tuple, optional
        Anonymous function(s) specifying this neuron's difference equation(s). Provided here since all neurons
        may utilize an explicit declaration of their update rule for numeric approximation purposes.

        This is strictly optional but necessary for utilizing Sapicore's integrators. If provided, the lambda syntax
        should be used (lambda x, data: f(x, data)).

        If the Euler update rule is v[t+1] = v[t] + f(v[t])*DT, `self.equation` should be the f(v[t]) portion.
        This is because RK4 needs to compute f(v[t]) at different points, e.g. half a time step forward.

    integrator: Integrator, optional
        Specifies an :class:`utils.integration.Integrator` to be used when approximating the next value(s)
        of this neuron's dynamic variable(s), e.g. voltage. Requires explicitly defining `equation`.
        Defaults to Runge-Kutta of order 4 to increase accuracy at a small performance penalty.

    input: Tensor
        PyTorch registered buffer tracking the input current(s) received in this simulation step.

    voltage: Tensor
        PyTorch registered buffer tracking the numeric state of the unit represented by this neuron instance,
        corresponding to a membrane voltage.

    kwargs:
        Additional instance attributes that the user may set.

    Warning
    -------
    When defining `equation` for a custom neuron model, the present value of `voltage` should NOT be added to the
    right hand side. Do NOT multiply by DT. These operations will be performed as part of the generic Euler forward.

    """

    _loggable_props_: tuple[str] = ("input", "voltage")

    # these instance attributes are registered as pytorch buffers.
    input: Tensor  # tensor storing input to the neuron.
    voltage: Tensor  # tensor storing membrane voltages.

    def __init__(self, equation: Callable = None, integrator: Integrator = RungeKutta(order=4), **kwargs):
        """Initializes generic instance attributes shared by all analog and spiking neuron derived classes."""
        # register universal attributes and tracking variables shared across components.
        super().__init__(**kwargs)

        # difference equation of this neuron model, to potentially be used with numeric approximation methods.
        self.equation = equation

        # integrators, though optional, should be incorporated into forward() if used at all.
        self.integrator = integrator

    def forward(self, data: Tensor) -> dict:
        """Passes input through the neuron.

        Parameters
        ----------
        data: Tensor
            Input current whose value is used to compute the next value of the numeric state tensor `voltage`.

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

    def integrate(self, **kwargs) -> Tensor:
        """Generic support for numeric approximation.

        This wrapper is meant to be called from within :meth:`~engine.neuron.Neuron.forward` in the voltage
        update step. Its purpose is to take the difference equation defining a neuron and approximate the
        voltage value at time t + :attr:`~utils.constants.DT`. Keyword arguments should include variables that
        `equation` depends on. The argument `x` is the tensor's state at time t, typically `voltage`.

        See Also
        --------
        :class:`~utils.integration.RungeKutta`

        """
        return self.integrator(x=self.voltage, equation=self.equation, **kwargs)

    def inject(self, current: Tensor):
        """Injects a current into this neuron.

        Parameters
        ----------
        current: Tensor
            Float tensor containing value(s) to be added to this neuron's `voltage` tensor.

        """
        self.voltage = self.voltage + current
