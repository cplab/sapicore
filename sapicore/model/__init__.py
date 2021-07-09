"""Model
========

A :class:`SapicoreModel` is a higher level abstraction of a
:class:`~sapicore.network.SapicoreNetwork` that can also contain
sub-:attr:`~SapicoreModel.networks` that are connected to each other via
neurons at the input/output boundaries of the sub-networks.

A :class:`SapicoreModel` additionally contains a learning element that can be
added and applied to the model.
"""
from typing import Dict, List, Optional, Tuple
from sapicore.network import SapicoreNetwork
from sapicore.neuron import SapicoreNeuron
from sapicore.learning import SapicoreLearning

__all__ = ('SapicoreModel', )


class SapicoreModel(SapicoreNetwork):
    """A Sapicore neuronal model instance.
    """

    networks: Dict[
        SapicoreNetwork,
        List[Tuple[SapicoreNetwork, Dict[SapicoreNeuron, SapicoreNeuron]]]] = {}
    """A dict mapping networks contained in the model to other networks
    in the model.

    Each value is a list of 2-tuples of ``(target_network, neurons)``.
    ``target_network`` is the network to which the source network in the key is
    connected. ``neurons`` is a dict whose keys and values are each a neuron
    from the source and target networks, respectively that connects the
    network.

    E.g. given ``network1`` and ``network2``, and neurons ``n1``, ``n2`` in
    ``network1`` that are respectively connected to neurons ``n3``, ``n4`` in
    ``network2``. Then :attr:`networks` is
    ``{network1: [(network2, {n1: n3, n2: n4})]}``.
    """

    def __init__(self, **kwargs):
        self.network = {}
        super(SapicoreModel, self).__init__(**kwargs)

    def add_learning_rule(
            self, name: str, learning_rule: SapicoreLearning) -> None:
        """Registers a learning rule that can be used by the model.
        You must implement :meth:`apply_learning` in order to concretely do
        something with the learning rule(s).

        The learning rule is also registered with pytroch as a "pytorch module"
        using ``add_module``. This enables usage of pytroch buffers by the
        learning rule instance.

        :param name: The name of the learning rule - required for pytorch.
        :param learning_rule: The :class:`~sapicore.learning.SapicoreLearning`.
        """
        if not isinstance(learning_rule, SapicoreLearning):
            raise TypeError(
                f'Only a SapicoreLearning can be added. '
                f'Cannot add <{learning_rule.__class__}>')

        self.add_module(name, learning_rule)

    def add_network(
        self, src_network: Optional[SapicoreNetwork], network_name: str,
            network: SapicoreNetwork,
            connected_neurons: Dict[SapicoreNeuron, SapicoreNeuron]) -> None:
        """Adds a named network to the model, optionally with a source network
        that connects to this network.

        The network is also registered with pytroch as a "pytorch module"
        using ``add_module``. This enables usage of pytroch buffers by the
        network instance.

        :param src_network: the optional source
            :class:`~sapicore.network.SapicoreNetwork`. May be None.
        :param network_name: The name of the network - required for pytorch.
        :param network: The :class:`~sapicore.network.SapicoreNetwork`.
        :param connected_neurons: A dict whose keys and values are
            :class:`~sapicore.neuron.SapicoreNeuron` instances that each map a
            neuron in the ``src_network`` to a neuron in the ``network``.
        """
        if not isinstance(network, SapicoreNetwork):
            raise TypeError(
                f'Only a SapicoreNetwork can be added. '
                f'Cannot add <{network.__class__}>')

        if src_network is not None:
            if not isinstance(src_network, SapicoreNetwork):
                raise TypeError(
                    f'Only a SapicoreNetwork can be added. '
                    f'Cannot add <{src_network.__class__}>')

            networks = self.networks[src_network]
            networks.append((network, connected_neurons))

        if network not in self.networks:
            self.networks[network] = []
            self.add_module(network_name, network)

    def initialize_state(self, **kwargs) -> None:
        super(SapicoreModel, self).initialize_state(**kwargs)
        for network in self.networks:
            network.initialize_state(**kwargs)

    def initialize_learning_state(self) -> None:
        """Initializes the learning rule.

        It must be overwritten to do any initialization.
        """
        pass

    def apply_learning(self, **kwargs) -> None:
        """Applies the learning rule to the model.

        It must be overwritten in user code to apply learning.
        """
        pass
