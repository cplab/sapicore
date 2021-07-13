"""Spiking neuron
=================
"""
from sapicore.neuron import SapicoreNeuron
from sapicore.neuron.spiking.integrate import Integration, IntegratorFactory

__all__ = ('SpikingNeuron', )


class SpikingNeuron(SapicoreNeuron):
    """Baseclass for spiking neurons.
    """

    _config_props_ = ('integrator_name', )

    integrator: Integration = None
    """The integrator to use for the neuron.

    It is automatically set during :meth:`post_config_applied` by looking up
    :attr:`integrator_name` in
    :class:`~sapicore.neuron.spiking.integrate.IntegratorFactory`.
    """

    integrator_name: str = 'runge_kutta'
    """The integration method to use. It must be one of those named in
    :class:`~sapicore.neuron.spiking.integrate.IntegratorFactory`.
    """

    def __init__(self, integrator_name='runge_kutta', **kwargs):
        super().__init__(**kwargs)

        self.integrator_name = integrator_name

    def post_config_applied(self):
        """Called post configuration as described in
        :class:`~tree_config.Configurable`.
        """
        super(SpikingNeuron, self).post_config_applied()
        self.integrator = IntegratorFactory.get_integrator_instance(
            self.integrator_name)
