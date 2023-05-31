from typing import Callable
from datetime import datetime as dt

import pytest
import torch

from sapicore.engine.ensemble.analog import AnalogEnsemble
from sapicore.engine.ensemble.spiking import LIFEnsemble, IZEnsemble
from sapicore.utils.integration import RungeKutta

import matplotlib.pyplot as plt

UNITS = 1
STEPS = 500
INTERVAL = 25
PLOT = False


class Curve:
    """Represents an ODE with an analytical solution, sets an initial condition `t0`,
    and computes its value `y0` at that point.

    Calling the object with an integer `t` will evaluate the curve at `t`.

    """

    def __init__(self, curve: Callable, ode: Callable, t0: float):
        self.curve = curve
        self.ode = ode
        self.t0 = t0
        self.y0 = self.curve(t=self.t0)

    def __call__(self, t: int):
        return self.curve(t=t)


class TestIntegration:
    @pytest.mark.functional
    def test_numeric_correctness(self):
        """Demonstrates correctness of Euler and RK4 using example ODEs with simple analytical solutions."""
        # define analytic curves, their respective ODEs, and t0 at which to start the simulation.
        curves = [
            Curve(curve=lambda t: torch.e**t, ode=lambda x: x, t0=1.0),
            Curve(curve=lambda t: (t**2.0) / 4.0, ode=lambda x: torch.sqrt(x), t0=4.0),
        ]

        for curve in curves:
            # initialize two analog ensembles that integrate using Euler/RK4.
            analog_euler = AnalogEnsemble(num_units=UNITS, integrator=RungeKutta(order=1), equation=curve.ode)
            analog_rk4 = AnalogEnsemble(num_units=UNITS, integrator=RungeKutta(order=4), equation=curve.ode)

            # update initial voltages to the value of the curve at its t0.
            analog_euler.voltage = torch.tensor(curve.y0)
            analog_rk4.voltage = torch.tensor(curve.y0)

            # compute the underlying function and Euler/RK4 approximations in the interval [0, INTERVAL).
            underlying = []
            euler = []
            rk4 = []

            for j in range(INTERVAL):
                # underlying curve at this time point.
                underlying.append(curve(t=curve.t0 + (j + 1.0) * analog_euler.dt))

                # integrate ensembles directly, bypassing forward().
                analog_euler.voltage = analog_euler.integrate()
                analog_rk4.voltage = analog_rk4.integrate()

                # store voltage values for the two integration modes.
                euler.append(analog_euler.voltage.mean().item())
                rk4.append(analog_rk4.voltage.mean().item())

            # plots show RK4 outperforming Euler for our test cases y(t) = e^t and y(t) = t^2 / 4.
            if PLOT:
                plt.plot(underlying)
                plt.plot(rk4)
                plt.plot(euler)
                plt.title("Ground Truth (Blue), RK4 (Orange), Euler (Green)")

                plt.show()

    @pytest.mark.parametrize("ref", [LIFEnsemble, IZEnsemble], ids=["LIF", "IZ"])
    @pytest.mark.functional
    def test_spiking_comparison(self, ref: type):
        """Shows that Euler/RK4 integration makes a difference for the default spiking neuron models (IZ, LIF)."""
        # initialize two spiking ensembles with Euler/RK4 integrators.
        euler = ref(identifier="Euler", num_units=UNITS, integrator=RungeKutta(order=1))
        rk4 = ref(identifier="RK4", num_units=UNITS, integrator=RungeKutta(order=4))

        # input current values were selected to trigger firing given the default LIF and IZ parameters.
        data = torch.zeros(UNITS) + (105.0 if ref is LIFEnsemble else 10.0)

        euler_values = torch.zeros((STEPS, UNITS))
        rk4_values = torch.zeros((STEPS, UNITS))

        start = dt.now()
        for i in range(STEPS):
            euler_values[i, :] = euler(data)["voltage"].cpu()
        euler_duration = dt.now() - start

        start = dt.now()
        for i in range(STEPS):
            rk4_values[i, :] = rk4(data)["voltage"].cpu()
        rk4_duration = dt.now() - start

        md = (euler_values - rk4_values).mean()
        sd = (euler_values - rk4_values).std()

        if PLOT:
            print(
                f"\nCompared to Euler, RK4 took x{rk4_duration / euler_duration}. "
                f"At DT={euler.dt}, Steps={STEPS}, Mean Euler-RK4 difference: {md:.2f} (SD={sd:.2f})"
            )

            plt.plot(euler_values.tolist())
            plt.plot(rk4_values.tolist())
            plt.title("Euler/RK4 Voltage Traces")
            plt.show()


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
