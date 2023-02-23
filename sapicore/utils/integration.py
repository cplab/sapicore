"""Integrator objects provide numerical methods for solving ODEs."""
from torch import Tensor

from typing import Callable
from sapicore.utils.constants import DT

__all__ = ("Integrator", "RungeKutta")


class Integrator:
    """Base integrator class.

    Parameters
    ----------
    identifier: str, optional
        Name of the integrator, e.g. "RK" for Runge-Kutta.

    order: int, optional
        Order of the integrator if applicable.

    step: float, optional
        Step size. Defaults to the simulation step :attr:`utils.constants.DT`.

    """

    def __init__(self, identifier: str = None, step: int = DT, **kwargs):
        self.identifier = identifier
        self.step = step

        # developer may override or define arbitrary attributes at instantiation.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, x: Tensor, equation: Callable, **kwargs) -> Tensor:
        """Approximates the function `equation`, whose keyword arguments are given as `kwargs`, given an
        initial value `x` and a time step `step` (defaults to the global :attr:`~utils.constants.DT`).

        Returns
        -------
        Tensor
            Approximate value(s) of the function at t0+DT, where t0 is the time `x` was obtained (current time).

        Raises
        ------
        NotImplementedError
            The __call__ method must be implemented by each derived integrator class.

        Warning
        -------
        Users are encouraged to use forward Euler (Runge-Kutta order 1) in their :meth:`~neuron.Neuron.forward`
        implementations, regardless of whether they anticipate using more advanced approximation methods.

        """
        raise NotImplementedError


class RungeKutta(Integrator):
    """Generic Runge-Kutta integrator.

    Parameters
    ----------
    identifier: str, optional
        Name of the integrator, e.g. "RK" for Runge-Kutta.

    order: int, optional
        Order of the integrator. Forward Euler (order 1) and RK4 are currently supported.

    References
    ----------
    `<https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html>`_
        :class:`~utils.integration.RungeKutta` implements the algorithm variants described in this tutorial.

    """

    def __init__(self, identifier: str = "RK", order: int = 1, **kwargs):
        super().__init__(identifier, **kwargs)
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
