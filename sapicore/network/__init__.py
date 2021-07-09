"""Network
===========

A network is composed of one or more :class:`~sapicore.neuron.SapicoreNeuron`
nodes that are linked to each other through a
:class:`~sapicore.synapse.SapicoreSynapse`. See
:meth:`~SapicoreNetwork.add_neuron` and :meth:`~SapicoreNetwork.add_synapse`.

See also :class:`~sapicore.model.SapicoreModel`.
"""
from typing import Dict, List, Optional
import torch.nn as nn
import torch

from sapicore.synapse import SapicoreSynapse
from sapicore.neuron import SapicoreNeuron
from tree_config import Configurable
from sapicore.logging import Loggable

__all__ = ('SapicoreNetwork', )


class SapicoreNetwork(Loggable, Configurable, nn.Module):
    """A Sapicore neuronal network instance.
    """

    neurons: Dict[SapicoreNeuron, List[SapicoreSynapse]] = {}
    """Dict whose keys are all the :class:`~sapicore.neuron.SapicoreNeuron` in
    the network and the corresponding values is a list of
    :class:`~sapicore.synapse.SapicoreSynapse` going out from the
    :class:`~sapicore.neuron.SapicoreNeuron`.
    """

    synapses: Dict[SapicoreSynapse, List[SapicoreNeuron]] = {}
    """Dict whose keys are all the :class:`~sapicore.synapse.SapicoreSynapse`
    in the network and the corresponding values is a list of
    :class:`~sapicore.neuron.SapicoreNeuron` into which the
    :class:`~sapicore.synapse.SapicoreSynapse` is terminating.
    """

    def __init__(self, **kwargs):
        super(SapicoreNetwork, self).__init__(**kwargs)
        self.neurons = {}
        self.synapses = {}

    def add_neuron(
            self, src_synapse: Optional[SapicoreSynapse], neuron_name: str,
            neuron: SapicoreNeuron) -> None:
        """Adds a named neuron to the network, optionally with a synapse
        terminating in the neuron.

        The neuron is also registered with pytroch as a "pytorch module"
        using ``add_module``. This enables usage of pytroch buffers by the
        neuron instance.

        :param src_synapse: the optional source
            :class:`~sapicore.synapse.SapicoreSynapse`. May be None.
        :param neuron_name: The name of the neuron - required for pytorch.
        :param neuron: The :class:`~sapicore.neuron.SapicoreNeuron`.
        """
        if not isinstance(neuron, SapicoreNeuron):
            raise TypeError(
                f'Only a SapicoreNeuron can be added. '
                f'Cannot add <{neuron.__class__}>')

        if src_synapse is not None:
            if not isinstance(src_synapse, SapicoreSynapse):
                raise TypeError(
                    f'Only a SapicoreSynapse can be added. '
                    f'Cannot add <{src_synapse.__class__}>')

            neurons = self.synapses[src_synapse]
            neurons.append(neuron)

        if neuron not in self.neurons:
            self.neurons[neuron] = []
            self.add_module(neuron_name, neuron)

    def add_synapse(
            self, src_neuron: Optional[SapicoreNeuron], synapse_name: str,
            synapse: SapicoreSynapse) -> None:
        """Adds a named synapse to the network, optionally with a neuron
        from which the synapse is exiting.

        The synapse is also registered with pytroch as a "pytorch module"
        using ``add_module``. This enables usage of pytroch buffers by the
        synapse instance.

        :param src_neuron: The optional
            :class:`~sapicore.neuron.SapicoreNeuron`. May be None.
        :param synapse_name: The name of the synapse - required for pytorch.
        :param synapse: The :class:`~sapicore.synapse.SapicoreSynapse`.
        """
        if not isinstance(synapse, SapicoreSynapse):
            raise TypeError(
                f'Only a SapicoreSynapse can be added. '
                f'Cannot add <{synapse.__class__}>')

        if src_neuron is not None:
            if not isinstance(src_neuron, SapicoreNeuron):
                raise TypeError(
                    f'Only a SapicoreNeuron can be added. '
                    f'Cannot add <{src_neuron.__class__}>')

            synapses = self.neurons[src_neuron]
            synapses.append(synapse)

        if synapse not in self.synapses:
            self.synapses[synapse] = []
            self.add_module(synapse_name, synapse)

    def initialize_state(self, **kwargs) -> None:
        """Initializes all the synapses and neurons in the network
        by calling their
        :meth:`~sapicore.synapse.SapicoreSynapse.initialize_state` and
        :meth:`~sapicore.neuron.SapicoreNeuron.initialize_state` methods.
        ``kwargs`` is passed on to synapses and neurons.
        """
        for neuron in self.neurons:
            neuron.initialize_state(**kwargs)

        for synapse in self.synapses:
            synapse.initialize_state(**kwargs)

    def forward(self, data: torch.tensor) -> torch.tensor:
        """The forward method, similar to pytorch models, that process the
        data through the network.
        """
        raise NotImplementedError
