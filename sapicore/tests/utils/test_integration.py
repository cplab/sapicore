import torch

from sapicore.engine.neuron.spiking.LIF import LIFNeuron
from sapicore.utils.integration import RungeKutta

from datetime import datetime as dt
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # TODO complete official pytests to further prove correctness.
    data = torch.zeros(1, device="cuda:0") + 125.0

    euler = LIFNeuron(integrator=RungeKutta(order=1), device="cuda:0")
    rk4 = LIFNeuron(integrator=RungeKutta(order=4), device="cuda:0")
    old = LIFNeuron(device="cuda:0")

    ret_euler = []
    ret_rk4 = []

    start = dt.now()
    for _ in range(1000):
        ret_euler += euler(data)["voltage"].cpu()
    print(f"Euler took {dt.now()-start}")

    start = dt.now()
    for _ in range(1000):
        ret_rk4 += rk4(data)["voltage"].cpu()
    print(f"RK4 took {dt.now() - start}")

    plt.plot(ret_euler)
    plt.plot(ret_rk4)

    plt.show()
