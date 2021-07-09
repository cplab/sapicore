"""Neuron integration
=====================

Provides integrators for a :class:`~sapicore.neuron.spkining.SpikingNeuron`.
"""
from typing import Dict, Type
import torch

__all__ = (
    'IntegratorFactoryBase', 'IntegratorFactory', 'Integration',
    'EulerIntegration', 'RungeKuttaIntegration',
)


class IntegratorFactoryBase:
    """Integration factory in which you can register and lookup integrators by
    name.
    """

    integrators: Dict[str, Type['Integration']] = {}
    """Dict of all registered named integrators.
    """

    @classmethod
    def register_integrator(
            cls, name: str, integrator_cls: Type['Integration']) -> None:
        """Registers a integrator class so it can be looked up woth ``name``.
        """
        cls.integrators[name] = integrator_cls

    def get_integrator_instance(self, name: str, **kwargs) -> 'Integration':
        """Looks up the integrator with the given ``name``, instantiates it
        with the given ``kwargs`` parameters and returns it.
        """
        return self.integrators[name](**kwargs)


class Integration:
    """Integrator base class that can integrate a
    :class:`~sapicore.neuron.spkining.SpikingNeuron`.
    """

    def integrate(self, neuron, x):
        """Integrates the neuron with the given data.
        """
        raise NotImplementedError


class EulerIntegration(Integration):
    """Euler integrator.
    """

    def integrate(self, neuron, x):
        # Integrate inputs.
        factor = (
            ((-neuron.volt_mem_prev + (x * neuron.r_mem)) / neuron.tc_decay) *
            neuron.dt)
        neuron.volt_mem = neuron.volt_mem_prev + (
                neuron.refrac_count == 0).float() * factor

        neuron.volt_mem_prev = neuron.volt_mem


class RungeKuttaIntegration(Integration):
    """Runge-Kutta integrator.
    """

    def integrate(self, neuron, x):
        # Integrate inputs.
        k_1 = (-neuron.volt_mem_prev + (x * neuron.r_mem)) / neuron.tc_decay
        k_2 = (-(neuron.volt_mem_prev + ((neuron.dt / 2) * k_1)) +
               (x * neuron.r_mem)) / neuron.tc_decay
        k_3 = ((-(neuron.volt_mem_prev + ((neuron.dt / 2) * k_2)) +
                (x * neuron.r_mem)) / neuron.tc_decay)
        k_4 = ((-(neuron.volt_mem_prev + (neuron.dt * k_3)) +
                (x * neuron.r_mem)) / neuron.tc_decay)

        factor = ((neuron.dt / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4))

        mask = (factor < 0)
        factor[mask] = 0

        neuron.volt_mem = neuron.volt_mem_prev + torch.mul((
            neuron.refrac_count == 0).float(), factor)

        neuron.volt_mem_prev = neuron.volt_mem


IntegratorFactory: IntegratorFactoryBase = IntegratorFactoryBase()
"""The integration factory with which integrators should be registered.
Sapicore defined integrators are automatically registered.
"""

IntegratorFactory.register_integrator('euler', EulerIntegration)
IntegratorFactory.register_integrator('runge_kutta', RungeKuttaIntegration)
