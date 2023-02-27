import pytest
import torch

from typing import Callable
from datetime import datetime as dt

from sapicore.engine.ensemble.analog import AnalogEnsemble
from sapicore.engine.ensemble.spiking import LIFEnsemble, IZEnsemble

from sapicore.utils.integration import RungeKutta
from sapicore.utils.constants import DT

import matplotlib.pyplot as plt

UNITS = 1
STEPS = 500


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
        """Demonstrate correctness of Euler and RK4 using example ODEs with simple analytical solutions."""
        # define analytical curves, their respective ODEs, start time and start value.
        curves = [
            Curve(lambda t: torch.e**t, lambda x: x, 1.0),
            Curve(lambda t: (t**2.0) / 4.0, lambda x: torch.sqrt(x), 4.0),
        ]

        for curve in curves:
            analog_euler = AnalogEnsemble(num_units=UNITS, integrator=RungeKutta(order=1), equation=curve.ode)
            analog_rk4 = AnalogEnsemble(num_units=UNITS, integrator=RungeKutta(order=4), equation=curve.ode)

            # set initial condition to e, the value of the analytical solution curve at t=1.
            analog_euler.voltage = torch.tensor(curve.y0)
            analog_rk4.voltage = torch.tensor(curve.y0)

            # compute differences between underlying function and euler/rk4 throughout simulation.
            underlying = []
            euler = []
            rk4 = []

            for j in range(25):
                # compute underlying curve at this time point.
                underlying.append(curve(t=curve.t0 + (j + 1.0) * DT))

                # integrate euler and rk4 ensembles.
                analog_euler.voltage = analog_euler.integrate()
                analog_rk4.voltage = analog_rk4.integrate()

                # store voltage values of euler and RK4 ensembles.
                euler.append(analog_euler.voltage.mean().item())
                rk4.append(analog_rk4.voltage.mean().item())

            # plots clearly show RK4 outperforming Euler.
            plt.plot(underlying)
            plt.plot(rk4)
            plt.plot(euler)
            plt.title("Underlying Function (Blue), RK4 (Orange), Euler (Green)")

            plt.show()

    @pytest.mark.parametrize("ref", [LIFEnsemble, IZEnsemble], ids=["LIF", "IZ"])
    @pytest.mark.functional
    def test_spiking_comparison(self, ref: type):
        # initialize LIF ensembles with two different approximation schemes.
        euler = ref(identifier="Euler", num_units=UNITS, integrator=RungeKutta(order=1))
        rk4 = ref(identifier="RK4", num_units=UNITS, integrator=RungeKutta(order=4))

        data = torch.zeros(UNITS) + (250.0 if isinstance(ref, LIFEnsemble) else 10.0)

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

        print(
            f"\nCompared to Euler, RK4 took x{rk4_duration/euler_duration}. "
            f"At DT={DT}, Steps={STEPS}, Euler-RK4 difference: {md:.2f} (SD={sd:.2f})"
        )

        plt.plot(euler_values.tolist())
        plt.plot(rk4_values.tolist())
        plt.title("Euler/RK4 Voltage Traces")
        plt.show()


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
