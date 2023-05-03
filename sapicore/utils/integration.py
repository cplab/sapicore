"""Integrator objects provide numerical methods for approximating ODE solutions."""
from torch import Tensor

from typing import Callable
from sapicore.utils.constants import DT

__all__ = ("Integrator", "RungeKutta")


class Integrator:
    """Base integrator class.

    Parameters
    ----------
    identifier: str, optional
        Name of the integrator.

    step: float, optional
        Step size. Defaults to :attr:`~utils.constants.DT`.

    Warning
    -------
    Users are encouraged to use the :class:`RungeKutta` object in their :meth:`~engine.neuron.Neuron.integrate`
    implementations (which in turn should be used in :meth:`~engine.neuron.Neuron.forward`), regardless of whether
    they anticipate using more advanced approximation methods.

    """

    def __init__(self, identifier: str = None, step: int = DT, **kwargs):
        self.identifier = identifier
        self.step = step

    def __call__(self, x: Tensor, equation: Callable, **kwargs) -> Tensor:
        """Approximates the next value of the ODE `equation`, whose keyword arguments are given as `kwargs`,
        for an initial value `x` and a time step `step` (defaults to the global :attr:`~utils.constants.DT`).

        Returns
        -------
        Tensor
            Approximate value(s) of the function at t+DT, where t is the time `x` was obtained (current time).

        Raises
        ------
        NotImplementedError
            The __call__ method must be implemented by each derived :class:`~utils.integration.Integrator`.

        """
        raise NotImplementedError


class RungeKutta(Integrator):
    """Generic Runge-Kutta integrator.

    When called, approximates the next value of an ODE `equation`, whose keyword arguments are given as `kwargs`,
    for an initial value `x` and a time step `step` (defaults to the global :attr:`~utils.constants.DT`),
    using the Runge-Kutta method of order `order`.

    Parameters
    ----------
    identifier: str, optional
        Name of the integrator.

    order: int, optional
        Order of the integrator. RK1 and RK4 are currently supported.

    Note
    ----
    The forward Euler method is Runge-Kutta of order 1.

    References
    ----------
    `Tutorial <https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html>`_
        :class:`~utils.integration.RungeKutta` implements the algorithm variants described above.

    """

    def __init__(self, identifier: str = "RK", order: int = 1, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        self.order = order

    def __call__(self, x: Tensor, equation: Callable, **kwargs) -> Tensor:
        """Uses the update rule `equation` to compute a Runge-Kutta approximation at t+DT.
        Variables needed by the difference equation are passed on by the caller as keyword arguments.

        Example
        -------
        Initialize LIF neuron specifying RK4 as the integrator:

            >>> from sapicore.engine.ensemble.spiking import LIFEnsemble
            >>> layer = LIFEnsemble(num_units=10, integrator=RungeKutta(order=4))

        Peek at the RK4 approximated value for `voltage` at t+DT (this will NOT modify the voltage value in-place):

            >>> data = Tensor(list(range(10)))
            >>> layer.integrate(data=data)

        The default LIF forward method makes use of the Integrator interface to update the voltage:

            >>> layer(data)
            >>> print(layer.voltage)

        """
        if self.order == 1:
            # standard forward Euler approximation.
            return x + equation(x=x, **kwargs) * self.step

        else:
            # compute RK4 slope estimates and return their weighted average.
            k1 = equation(x=x, **kwargs)
            k2 = equation(x=x + (self.step / 2.0) * k1, **kwargs)
            k3 = equation(x=x + (self.step / 2.0) * k2, **kwargs)
            k4 = equation(x=x + self.step * k3, **kwargs)

            return x + (self.step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
