""" Networks are graph representations of neuron ensembles connected by synapses. """
import os
import logging

import networkx as nx
from networkx import DiGraph

import torch
from torch.nn import Module

from sapicore.engine.neuron import Neuron
from sapicore.engine.synapse import Synapse

from sapicore.utils.io import parse_yaml
from sapicore.utils.signals import flatten
from sapicore.utils.constants import ROOT

__all__ = ("Network",)


class Network(Module):
    """Generalized spiking neural network instance. Maintains and orchestrates the processing of multiple
    ensembles connected by synapses.

    Networks may be constructed from a configuration dictionary (read from a YAML or defined programmatically)
    by applying :meth:`build_network`. User may also initialize an empty network and apply :meth:`add_ensembles`
    and :meth:`add_synapses` to build the network `graph` directly. One or more `root` nodes may be specified
    in the YAML/dictionary or passed to the Network constructor.

    Parameters
    ----------
    identifier: str, optional
        String identifier for file I/O and visualization operations.

    configuration: dict, optional
        Configuration dictionary containing model specification and simulation parameters. Used to construct
        the `graph` and initialize the `root` instance attributes if provided.

    device: str
        Specifies a hardware device on which to process this object's tensors, e.g. "cuda:0".
        Must be a valid torch.device() string value. Defaults to "cpu".

    graph: networkx.DiGraph, optional
        Initialize a network object with a ready-made graph representation containing ensemble and synapse objects,
        which are themselves :mod:`torch.nn.Module` and support operations such as autograd.

    Notes
    -----
    Synapse objects know their source and destination ensemble references; they are 2D objects
    capable of connecting two ensembles in a unidirectional fashion with heterogeneous parameters.
    Lateral connections from an ensemble to itself are supported by design and warrant no special treatment.

    On :meth:`forward` calls, the nodes of the graph are traversed in breadth-first search (BFS) order. The ensemble
    referenced by the current node is processed first, followed by its outgoing synapses. If multiple source nodes
    exist, the ``networkX`` multi-source BFS implementation is used.

    """

    def __init__(
        self, identifier: str = None, device: str = "cpu", configuration: dict = None, graph: DiGraph = None, **kwargs
    ):
        super().__init__()

        # model-related common instance attributes.
        self.identifier = identifier
        self.configuration = configuration
        self.device = device

        self.graph = DiGraph() if not graph else graph
        self.roots = None

        # simulation-related common instance attributes.
        self.simulation_step = 0
        self.traversal_order = []

        # developer may override or define arbitrary attributes at instantiation.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # network construction from configuration file overrides programmatic initialization.
        if configuration:
            self.build()

        # automatically mark root nodes by their in-degree.
        self._find_roots()

    def add_ensembles(self, *args: Neuron | dict | str):
        """Adds ensemble nodes to the network graph from a list of string `path`s to YAML dictionaries,
        from pre-initialized :class:`neuron.Neuron` objects or children thereof (e.g., any `Ensemble`),
        or from configuration dictionaries. User may mix and match if necessary.

        Raises
        ------
        TypeError
            If user attempts to pass an argument that is not a string path, a dictionary, or an instance derived
            from :class:`neuron.Neuron`.

        Warning
        -------
        The component `identifier` field is used to reference nodes and edge sources or destinations when
        constructing the network graph.

        """
        for ensemble in args:
            if isinstance(ensemble, Neuron):
                # handle the object arguments.
                self.graph.add_node(ensemble.identifier, reference=ensemble, device=self.device)

            elif isinstance(ensemble, dict):
                ensemble, comp_cfg = self._object_from_configuration(cfg=ensemble, comp_type="ensemble")
                self.graph.add_node(ensemble.identifier, reference=ensemble, device=self.device)

            elif isinstance(ensemble, str):
                if os.path.exists(os.path.join(ROOT, ensemble)):
                    # add the ensemble reference to the graph as a node.
                    ensemble, comp_cfg = self._object_from_configuration(
                        path=os.path.join(ROOT, ensemble), comp_type="ensemble"
                    )
                    self.graph.add_node(ensemble.identifier, reference=ensemble, device=self.device)
                else:
                    logging.info(f"Could not add ensemble, invalid path given: {ensemble}")
                    continue
            else:
                raise TypeError("Ensembles must be given as object references, dictionaries, or paths to YAML.")

            # heterogenize modifies the object iff a `sweep` specification was provided in the model configuration.
            ensemble.heterogenize()

        # update root node list.
        self._find_roots()

    def add_synapses(self, *args: Synapse | dict | str):
        """Adds synapse edges to the network graph from a list of string `path`s to YAML dictionaries,
        from pre-initialized :class:`synapse.Synapse` objects or children thereof (e.g., `STDPSynapse`),
        or from configuration dictionaries. User may mix and match if necessary.

        Raises
        ------
        TypeError
            If user attempts to pass an argument that is not a string path, a dictionary, or an instance derived
            from :class:`synapse.Synapse`.

        """
        for synapse in args:
            if isinstance(synapse, Synapse):
                # handle the object arguments.
                self.graph.add_edge(synapse.src_ensemble.identifier, synapse.dst_ensemble.identifier, reference=synapse)

            elif type(synapse) is dict:
                synapse, comp_cfg = self._object_from_configuration(cfg=synapse, comp_type="synapse")
                self.graph.add_edge(synapse.src_ensemble.identifier, synapse.dst_ensemble.identifier, reference=synapse)

            elif type(synapse) is str:
                if os.path.exists(os.path.join(ROOT, synapse)):
                    # add the synapse reference to the graph as a node.
                    synapse, comp_cfg = self._object_from_configuration(
                        path=os.path.join(ROOT, synapse), comp_type="synapse"
                    )
                    self.graph.add_edge(*synapse.identifier.split("->"), reference=synapse)
                else:
                    logging.info(f"Could not add synapse, invalid path given: {synapse}")
                    continue
            else:
                raise TypeError("Synapses must be given as object references, dictionaries, or paths to YAML.")

            # # heterogenize modifies the object iff a `sweep` specification was provided in the model configuration.
            synapse.heterogenize()

        # update root node list.
        self._find_roots()

    def get_ensembles(self):
        """Returns all ensemble references in the network `graph`."""
        return [self.graph.nodes[i]["reference"] for i in self.graph.nodes]

    def get_synapses(self):
        """Returns all synapse references in the network `graph`."""
        return [self.graph.edges[s, d]["reference"] for s, d in self.graph.edges]

    def build(self):
        """Adds and initializes named components to the graph from this instance's `configuration` dictionary.

        Note
        ----
        Network YAMLs reference ensemble and synapse YAMLs by their path relative to that specified in the `root`
        field (or the project root if that field was not provided).

        """
        # set network identifier string from configuration file.
        self.identifier = self.configuration.get("identifier", "net")

        # use network API to add ensembles and synapses based on configuration dictionary or YAML paths.
        if "ensembles" in self.configuration.get("model"):
            ensemble_paths = [
                os.path.join(self.configuration.get("root", ""), ep)
                for ep in self.configuration.get("model", {}).get("ensembles")
            ]
            self.add_ensembles(*ensemble_paths)

        if "synapses" in self.configuration.get("model"):
            synapse_paths = [
                os.path.join(self.configuration.get("root", ""), sp)
                for sp in self.configuration.get("model", {}).get("synapses")
            ]
            self.add_synapses(*synapse_paths)

    # To micromanage the forward/backward sweeps, subclass Network and override summation(), forward(), backward().
    @staticmethod
    def summation(synaptic_input: list[torch.tensor]) -> torch.tensor:
        """Adds up inputs from multiple synapse objects onto the same ensemble, given as rows.

        Note
        ----
        If your model requires some preprocessing of inputs to the postsynaptic neuron, it can be implemented
        by overriding this method.

        """
        return torch.sum(torch.vstack(synaptic_input), dim=0)

    def backward(self) -> None:
        """Processes a backward sweep for this network object.

        This is not required for SNNs that learn with STDP and can be implemented by child classes that
        might require a backward pass, e.g. DNN variants.

        Warning
        -------
        You are encouraged to take advantage of well-established libraries for declaring and instantiating classic
        DNN variants that learn with backpropagation. This simulation engine is SNN-centric and was designed
        and optimized for that use case.

        """
        pass

    def forward(self, data: torch.tensor) -> None:
        """Processes current simulation step for this network object.

        In this generic implementation, forward call order is determined by a BFS traversal starting from
        the edges fanning out of the root nodes. Every node is visited once; for each node, every outgoing synapse
        is forwarded. That way, we are guaranteed to forward any component exactly once on every simulation step.

        Parameters
        ----------
        data: torch.tensor
            External input to be processed by this generic network.

        Warning
        -------
        Networks now support multiple root nodes and external input streams. In those cases, the graph is
        traversed layer-wise, using multi-source BFS.

        See Also
        --------
        :meth:`simulation.Simulator.run`
            For a demonstration of how to programmatically feed heterogeneous external current to multiple nodes.
            Note that this functionality will eventually be relegated to the data loader, with arbitrary current
            injections during the simulation being implemented by biases.

        """
        # follow outgoing synapses between vertices, forwarding the entire network on each iteration.
        for i, ensemble in enumerate(self.traversal_order):
            # list all synapses coming in and out of this ensemble. Networkx references it by its string identifier.
            incoming_synapses = self._in_edges(ensemble)
            outgoing_synapses = self._out_edges(ensemble)

            # shortcut to actual ensemble reference.
            ensemble_ref = self.graph.nodes[ensemble].get("reference")

            if ensemble_ref.identifier in self.roots:
                # if this is a root node, treat stream from data loader as inbound synaptic input.
                integrated_data = data[self.roots.index(ensemble_ref.identifier)]
            else:
                # apply a summation function to synaptic data flowing into this ensemble (torch.sum by default).
                integrated_data = self.summation([synapse.output for synapse in incoming_synapses]).to(self.device)

            # forward current ensemble.
            ensemble_ref(integrated_data)

            # use current ensemble's analog or spike data to forward its outgoing synapses.
            if hasattr(ensemble_ref, "spiked"):
                [synapse(ensemble_ref.spiked.float()) for synapse in outgoing_synapses]

            elif hasattr(ensemble_ref, "voltage"):
                [synapse(ensemble_ref.voltage) for synapse in outgoing_synapses]

        self.simulation_step += 1

    def _in_edges(self, node: str) -> list[Synapse]:
        """Returns list of synapse objects going into the ensemble `node`."""
        return [ref.get("reference") for _, _, ref in list(self.graph.in_edges(node, data=True))]

    def _out_edges(self, node: str) -> list[Synapse]:
        """Returns list of synapse objects fanning out of the ensemble `node`."""
        return [ref.get("reference") for _, _, ref in list(self.graph.out_edges(node, data=True))]

    def _find_roots(self):
        """Locates one or more node identifiers, corresponding to :class:`~neuron.Neuron` or a child class thereof
        (including ensembles), that are exposed to external input. Those are used as starting points for the multi-BFS
        forward sweep, whose order also gets updated every time this method is called.
        """
        # networkx has wrong type hinting for in_degree's return value, but it can be iterable (not always int).
        self.roots = [node for node, degree in self.graph.in_degree() if degree == 0]

        # breadth-first search (BFS) iterator used to traverse the network graph layer by layer.
        # returns a list of layers, first one consisting of the root nodes. Flatten and visit sequentially.
        self.traversal_order = flatten(list(nx.bfs_layers(self.graph, self.roots)))

    def _object_from_configuration(
        self, path: str = None, cfg: dict = None, comp_type: str = None
    ) -> (Neuron | Synapse, dict):
        """Constructs an objects from the YAML in `path` or from `cfg` dictionary if provided."""
        # load configuration dictionary from ensemble descriptor YAML.
        if path and not cfg:
            cfg = parse_yaml(path)

        # initialize the ensemble/synapse based on its class reference, given in its configuration file.
        temp_import = __import__(cfg["package"], globals(), locals(), ["*"])
        class_ref = getattr(temp_import, cfg["class"])

        if "ensemble" in comp_type:
            comp_ref = class_ref(**cfg["model"], configuration=cfg, device=self.device)

        else:
            comp_ref = class_ref(
                **cfg["model"],
                configuration=cfg,
                device=self.device,
                src_ensemble=self.graph.nodes[cfg["model"]["source"]]["reference"],
                dst_ensemble=self.graph.nodes[cfg["model"]["target"]]["reference"],
            )

        # pass autograd setting from network configuration file to the synapse, if provided.
        if cfg.get("autograd"):
            cfg["model"]["autograd"] = self.cfg["autograd"]

        return comp_ref, cfg
