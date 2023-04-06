""" Networks are graph representations of neuron ensembles connected by synapses. """
import os

import networkx as nx
from networkx import DiGraph

import torch
from torch.nn import Module

from sapicore.engine.component import Component
from sapicore.engine.neuron import Neuron
from sapicore.engine.synapse import Synapse

from sapicore.utils.io import DataAccumulatorHook, flatten, load_yaml

__all__ = ("Network",)


class Network(Module):
    """Generalized spiking neural network instance. Maintains and orchestrates the processing of multiple
    ensembles connected by synapses.

    Networks may be constructed from a configuration dictionary (read from a YAML or defined programmatically)
    by applying :meth:`build`. User may also initialize an empty network and apply :meth:`add_ensembles`
    and :meth:`add_synapses` to build the network `graph` directly. One or more `root` nodes may be specified
    in the YAML/dictionary or passed to the Network constructor.

    Parameters
    ----------
    identifier: str, optional
        String identifier for file I/O and visualization operations.

    configuration: dict, optional
        Configuration dictionary containing model specification and simulation parameters. Used to construct
        the `graph` and initialize the `root` instance attributes if provided.

    roots: list of str, optional
        List of one or more ensemble identifiers corresponding to root nodes, i.e. the starting point of each
        forward pass. Can be passed to the constructor, configured in a YAML (model/roots), or automatically
        detected by checking for nodes with in-degree=0 (the latter is a fallback strategy and does not override
        user configuration). In cases where multiple external data tensors are passed to
        :meth:`~engine.network.Network.forward` or :meth:`~model.Model.fit`, they are routed to these ensembles
        (the i-th tensor to the i-th root by convention).

    device: str
        Specifies a hardware device on which to process this object's tensors, e.g. "cuda:0".
        Must be a valid torch.device() string value. Defaults to "cpu".

    Note
    ----
    Synapse objects know their source and destination ensemble references; they are 2D objects
    capable of connecting two ensembles in a unidirectional fashion with heterogeneous parameters.
    Lateral connections from an ensemble to itself are supported by design and warrant no special treatment.

    On :meth:`forward` calls, the nodes of the graph are traversed in breadth-first search (BFS) order. The ensemble
    referenced by the current node is processed first, followed by its outgoing synapses. If multiple source nodes
    exist, the ``networkX`` multi-source BFS implementation is used.

    """

    def __init__(
        self, identifier: str = None, device: str = "cpu", configuration: dict = None, roots: list[str] = None
    ):
        super().__init__()

        # model-related common instance attributes.
        self.identifier = identifier
        self.configuration = configuration
        self.device = device
        self.graph = DiGraph()

        # if provided, ensure user's root node selection is preserved.
        self.roots = [] if roots is None else roots
        self.root_lock = True if self.roots else False

        # simulation-related common instance attributes.
        self.simulation_step = 0
        self.traversal_order = []

        # store identifiers of registered components for lookup purposes.
        self._ensemble_identifiers = []
        self._synapse_identifiers = []

        # network construction from configuration file overrides programmatic initialization.
        if configuration:
            self._build()

        # if still not set, automatically mark root nodes by their in-degree.
        if not self.root_lock:
            # keep unlocked in case of subsequent `add_ensemble` or `add_synapse` calls.
            self._find_roots()

    def __getitem__(self, item: str) -> Component | None:
        """Look up and return a network component by its string identifier."""
        if self.graph.nodes.get(item):
            return self.graph.nodes.get(item)["reference"]

        elif self.graph.edges.get(item.split("->")):
            return self.graph.edges.get(item.split("->"))["reference"]

        else:
            raise KeyError(f"Component {item} does not exist in this network.")

    def __str__(self):
        """Describe the network to the user."""
        num_neurons = sum([self.graph.nodes[i]["reference"].num_units for i in self.graph.nodes])

        total_synapses = int(sum([s[0] * s[1] for s in [i.matrix_shape for i in self.get_synapses()]]))
        active_synapses = int(sum([conn.sum() for conn in [i.connections for i in self.get_synapses()]]))

        return (
            f"'{self.identifier}': {len(self.graph.nodes)} layers, {num_neurons} neurons, "
            f"{len(self.graph.edges)} synapse matrices with {active_synapses}/{total_synapses} active connections."
        )

    def add_ensembles(self, *args: Neuron | dict | str):
        """Adds ensemble nodes to the network graph from paths to YAML, from pre-initialized
        :class:`engine.neuron.Neuron` objects or children thereof (e.g., any `Ensemble`), or from
        configuration dictionaries. User may mix and match if necessary.

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
            if isinstance(ensemble, str):
                if os.path.exists(ensemble):
                    # instantiate the ensemble from configuration file.
                    ensemble, _ = self._object_from_configuration(path=ensemble, comp_type="ensemble")
                else:
                    raise IOError(f"Could not add ensemble, invalid path given: {ensemble}")

            elif isinstance(ensemble, dict):
                ensemble, _ = self._object_from_configuration(cfg=ensemble, comp_type="ensemble")

            # verify that `ensemble` now references a valid object derived from Neuron.
            if not isinstance(ensemble, Neuron):
                raise TypeError("Ensembles must be given as object references, dictionaries, or paths to YAML.")

            # add ensemble to the network graph and register its identifier.
            if self._unique_name(ensemble.identifier):
                self.graph.add_node(ensemble.identifier, reference=ensemble)
                self._ensemble_identifiers.append(ensemble.identifier)
            else:
                raise ValueError(f"Ensemble named {ensemble.identifier} already exists in this network.")

            # heterogenize modifies the object iff a `sweep` specification was provided in the model configuration.
            ensemble.heterogenize(num_combinations=ensemble.num_units)

        # update root node list.
        if not self.root_lock:
            self._find_roots()

    def add_synapses(self, *args: Synapse | dict | str):
        """Adds synapse edges to the network graph from paths to YAML, from pre-initialized
        :class:`engine.synapse.Synapse` objects or children thereof (e.g.,`STDPSynapse`), or from
        configuration dictionaries. User may mix and match if necessary.

        Raises
        ------
        TypeError
            If user attempts to pass an argument that is not a string path, a dictionary, or an instance derived
            from :class:`synapse.Synapse`.

        """
        for synapse in args:
            if isinstance(synapse, str):
                # instantiate synapse from configuration file.
                if os.path.exists(synapse):
                    # add the synapse reference to the graph as a node.
                    synapse, _ = self._object_from_configuration(path=synapse, comp_type="synapse")
                else:
                    raise IOError(f"Could not add synapse, invalid path given: {synapse}")

            elif isinstance(synapse, dict):
                # instantiate synapse from dictionary.
                synapse, _ = self._object_from_configuration(cfg=synapse, comp_type="synapse")

            # verify that `synapse` now references a valid Synapse object and has source and destination ensembles.
            if isinstance(synapse, Synapse):
                if not (synapse.src_ensemble and synapse.dst_ensemble):
                    raise AttributeError(
                        f"Synapse {synapse.identifier} is missing a source and/or a destination "
                        f"ensemble attribute(s)."
                    )

                # add synapse to network graph and register its identifier.
                if self._unique_name(synapse.identifier):
                    self.graph.add_edge(
                        synapse.src_ensemble.identifier, synapse.dst_ensemble.identifier, reference=synapse
                    )
                    self._synapse_identifiers.append(synapse.identifier)
                else:
                    raise ValueError(f"Synapse named {synapse.identifier} already exists in this network.")

            else:
                raise TypeError("Synapses must be given as object references, dictionaries, or paths to YAML.")

            # # heterogenize modifies the object iff a `sweep` specification was provided in the model configuration.
            synapse.heterogenize(num_combinations=synapse.num_units)

        # update root node list.
        if not self.root_lock:
            self._find_roots()

    def get_ensembles(self):
        """Returns all ensemble references in the network `graph`."""
        return [self.graph.nodes[i]["reference"] for i in self.graph.nodes]

    def get_synapses(self):
        """Returns all synapse references in the network `graph`."""
        return [self.graph.edges[s, d]["reference"] for s, d in self.graph.edges]

    def _build(self):
        """Adds and initializes named components to the graph from this instance's `configuration` dictionary.

        Note
        ----
        Network YAMLs reference ensemble and synapse YAMLs by their path relative to that specified in the `root`
        field (or the project root if that field was not provided).

        Warning
        -------
        This function is only called once during instantiation from a configuration file.
        Do not call it from external modules.

        """
        # set network identifier string from configuration file.
        self.identifier = self.configuration.get("identifier", "")

        # avoid repeatedly recomputing the roots while the network is being automatically built.
        self.root_lock = True

        # use network API to add ensembles and synapses based on configuration dictionary or YAML paths.
        if "ensembles" in self.configuration.get("model", {}):
            ensemble_paths = [
                os.path.join(self.configuration.get("root", ""), ep)
                for ep in self.configuration.get("model", {}).get("ensembles", ())
            ]
            self.add_ensembles(*ensemble_paths)

        if "synapses" in self.configuration.get("model", {}):
            synapse_paths = [
                os.path.join(self.configuration.get("root", ""), sp)
                for sp in self.configuration.get("model", {}).get("synapses", ())
            ]
            self.add_synapses(*synapse_paths)

        # set roots from configuration if key `roots` exists and roots were not set yet.
        config_roots = self.configuration.get("model", {}).get("roots", False)

        if config_roots:
            # update root nodes from configuration and lock s.t. it does not get recomputed.
            self.roots = config_roots

        # this will only be arrived at if user hasn't passed or configured `roots`.
        # note that self.roots is initialized to [] to prevent BFS failing on empty root list.
        elif not self.roots:
            # automatically mark nodes with in-degree == 0 as roots.
            self._find_roots()

            # unlock to allow automatic root recomputing in subsequent `add_ensembles` or `add_synapses` calls.
            self.root_lock = False

    def add_data_hook(self, data_dir: str, steps: int, *args: Component) -> list:
        """Attach a data accumulator forward hook to some or all network components.

        Parameters
        ----------
        data_dir: str
            Path to directory in which to save intermediate simulation output.

        steps: int
            Total number of simulation steps in the experiment to be logged.
            Required for HDF5 sizing and chunk management.

        args: Component, optional
            Components to attach data hooks to. If not provided, data will be logged for all components.

        """
        hooks = []
        if not args:
            for ensemble in self.get_ensembles():
                hooks.append(DataAccumulatorHook(ensemble, data_dir, ensemble.loggable_props, steps))

            for synapse in self.get_synapses():
                hooks.append(DataAccumulatorHook(synapse, data_dir, synapse.loggable_props, steps))
        else:
            for comp in args:
                hooks.append(DataAccumulatorHook(comp, data_dir, comp.loggable_props, steps))

        return hooks

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
        might require a backward pass.

        """
        pass

    def forward(self, data: torch.tensor) -> dict:
        """Processes current simulation step for this network object.

        In this generic implementation, forward call order is determined by a BFS traversal starting from
        the edges fanning out of the root nodes. Every node is visited once; for each node, every outgoing synapse
        is forwarded. That way, we are guaranteed to forward any component exactly once on every simulation step.

        Parameters
        ----------
        data: torch.tensor
            External input to be processed by this generic network.

        Returns
        -------
        dict
            Dictionary containing the loggable properties of this network's ensembles and synapses in this
            simulation step.

        Warning
        -------
        Networks now support multiple root nodes and external input streams. In those cases, the graph is
        traversed layer-wise, using multi-source BFS.

        See Also
        --------
        :meth:`simulation.GenericSimulator.run`
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
            ensemble_ref = self.graph.nodes[ensemble]["reference"]

            if ensemble_ref.identifier not in self.roots:
                # apply a summation function to synaptic data flowing into this ensemble (torch.sum by default).
                if incoming_synapses:
                    integrated_data = self.summation([synapse.output for synapse in incoming_synapses]).to(self.device)
                else:
                    integrated_data = ensemble_ref.input

            else:
                # if this is a root node, treat its stream from the data loader as inbound synaptic input.
                external = [data[self.roots.index(ensemble_ref.identifier)]] if isinstance(data, list) else [data]
                feedback = [synapse.output for synapse in incoming_synapses]

                integrated_data = self.summation(external + feedback)

            # forward current ensemble.
            ensemble_ref(integrated_data)

            # use the ensemble's declared output field (voltage or spiked) to forward its outgoing synapses.
            [synapse(getattr(ensemble_ref, ensemble_ref.output_field).float()) for synapse in outgoing_synapses]

        self.simulation_step += 1

        # build and pass a state dictionary for use by calling module.
        state = {"ensembles": {}, "synapses": {}}
        for comp in self.get_ensembles():
            state["ensembles"][comp.identifier] = comp.loggable_state()

        for comp in self.get_synapses():
            state["synapses"][comp.identifier] = comp.loggable_state()

        return state

    def _in_edges(self, node: str) -> list[Synapse]:
        """Returns list of synapse objects going into the ensemble `node`."""
        if node in list(self.graph.nodes):
            return [ref["reference"] for _, _, ref in list(self.graph.in_edges(node, data=True))]

    def _out_edges(self, node: str) -> list[Synapse]:
        """Returns list of synapse objects fanning out of the ensemble `node`."""
        if node in list(self.graph.nodes):
            return [ref["reference"] for _, _, ref in list(self.graph.out_edges(node, data=True))]

    def _unique_name(self, candidate: str) -> bool:
        """Checks whether a component identifier is identical to one that has already been added to this network.

        Parameters
        ----------
        candidate: str
            Name of new component to be added.

        Returns
        -------
        bool
            True if the name is unique, False otherwise.

        """
        return not (candidate in self._synapse_identifiers + self._ensemble_identifiers)

    def _find_roots(self):
        """Identifies nodes that have no incoming edges. Those are used as starting points for the multi-BFS
        forward sweep, whose order also gets updated every time this method is called.

        Warning
        -------
        Roots are updated on network initialization and on every application of :meth:`add_ensemble` and
        :meth:`add_synapse`. If adding a synapse results in a rootless recurrent network, all nodes will have
        in-degree > 0. In such cases, the method will mark the first ensemble added `self.get_ensembles()[0]`
        as the root and direct external input to it. This behavior can be modified by setting `self.roots`
        directly after building the network.

        """
        updated_roots = [node for node, degree in self.graph.in_degree() if degree == 0]
        if updated_roots:
            self.roots = updated_roots

        elif self.get_ensembles():
            # if the network became rootless after adding a synapse, default to the first ensemble as the root.
            self.roots = [self.get_ensembles()[0].identifier]

        # breadth-first search (BFS) iterator used to traverse the network graph layer by layer.
        # returns a list of layers, first one consisting of the root nodes. Flatten and visit sequentially.
        self.traversal_order = flatten(list(nx.bfs_layers(self.graph, self.roots)))

    def _object_from_configuration(
        self, path: str = None, cfg: dict = None, comp_type: str = None
    ) -> (Neuron | Synapse, dict):
        """Constructs an objects from the YAML in `path` or from `cfg` dictionary if provided."""
        # load configuration dictionary from ensemble descriptor YAML.
        if path and not cfg:
            cfg = load_yaml(path)

        # initialize the ensemble/synapse based on its class reference, given in its configuration file.
        temp_import = __import__(cfg["package"], globals(), locals(), ["*"])
        class_ref = getattr(temp_import, cfg["class"])

        if comp_type == "ensemble":
            comp_ref = class_ref(**cfg["model"], configuration=cfg, device=self.device)

        else:
            comp_ref = class_ref(
                **cfg["model"],
                configuration=cfg,
                device=self.device,
                src_ensemble=self.graph.nodes[cfg["model"]["source"]]["reference"],
                dst_ensemble=self.graph.nodes[cfg["model"]["target"]]["reference"],
            )

        return comp_ref, cfg
