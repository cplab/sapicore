""" Simulate a network from a configuration YAML. """
import os
import logging

from argparse import ArgumentParser
from datetime import datetime

import torch
import networkx as nx
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from sapicore.engine.network import Network
from sapicore.pipeline import Pipeline

from sapicore.utils.seed import set_seed
from sapicore.utils.signals import design_input_current
from sapicore.utils.io import DataAccumulatorHook, ensure_dir, dump_yaml, log_settings
from sapicore.utils.constants import DT, TIME_FORMAT, SEED
from sapicore.utils.tensorboard import TensorboardWriter, HDFData

from sapicore.tests import ROOT

# default hardware device for this pipeline.
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda:0"


class Simulator(Pipeline):
    """The default simulation pipeline, bringing together model specification, disk I/O, and visualization.
    Initialize a network and configure the simulation by providing a dictionary or a path to a YAML.

    Parameters
    ----------
    configuration: dict or str
        Configuration dictionary or path to a configuration YAML. If the path provided is relative, the simulation
        pipeline will assume that the root is sapicore/tests, in order to support execution of YAML-configured tests.

    """

    def __init__(self, configuration: dict | str = None, **kwargs):
        super().__init__(configuration=configuration, **kwargs)

        # set random seed for entire simulation and apply default logger configuration.
        set_seed(SEED)
        log_settings()

    def run(self):
        """Run the simulation based on the given configuration.

        Raises
        ------
        KeyError
            If full path to project `root` directory is not provided as a key in the configuration YAML or dictionary.

        """
        # get full path to project root directory from configuration file.
        root = self.configuration.get("root")
        if not root:
            raise KeyError('Configuration missing project "root" key.')

        elif not os.path.isabs(root):
            # if full path not provided, the relative path is appended to sapinet/tests.
            root = os.path.join(ROOT, root)

        # create timestamped run directory.
        stamp = datetime.now().strftime(TIME_FORMAT)
        run_dir = ensure_dir(os.path.join(root, stamp))

        # save a copy of the configuration to this run's directory for reference--need to handle dict-only case.
        dump_yaml(self.configuration, os.path.join(run_dir, self.configuration.get("identifier", "run_cfg") + ".yaml"))

        # load simulation parameters from configuration dictionary.
        steps = self.configuration.get("simulation", {}).get("steps")
        disk = self.configuration.get("simulation", {}).get("disk")
        tensorboard = self.configuration.get("simulation", {}).get("tensorboard")

        # create data and tensorboard subdirectories and define a tensorboard writer object.
        tb_dir = ensure_dir(os.path.join(run_dir, "tb"))
        data_dir = ensure_dir(os.path.join(run_dir, "data"))

        # initialize network object with given configuration.
        net = Network(configuration=self.configuration, device=DEVICE)

        # save basic networkx graph plot showing ensembles and their general connectivity patterns.
        nx.draw(net.graph, node_size=500, with_labels=True, pos=nx.kamada_kawai_layout(net.graph))
        plt.savefig(fname=os.path.join(run_dir, net.identifier + ".svg"))

        # set up data accumulator hooks if required. Note that they are automatically applied on forward calls.
        if disk:
            for ensemble in net.get_ensembles():
                DataAccumulatorHook(ensemble, data_dir, ensemble._loggable_props_, steps)

            for synapse in net.get_synapses():
                DataAccumulatorHook(synapse, data_dir, synapse._loggable_props_, steps)

        # design input current time series for each root ensemble exposed to external input.
        net_roots = [net.graph.nodes[i]["reference"] for i in net.roots]

        currents = [torch.zeros(size=(r.num_units, steps), dtype=torch.float, device=DEVICE) for r in net_roots]
        recipe = self.configuration.get("simulation", {}).get("current")

        # if user provided a current recipe in the YAML, parse it.
        if recipe:
            for i, root_node in enumerate(net.roots):
                # extract recipe for this root node.
                temp = torch.tensor(recipe.get(root_node))
                if temp.dim() == 1:
                    # if specified as a 1D tensor, feed same current to all neurons in the tensor.
                    currents[i] = design_input_current(temp, steps, root_node.num_units)
                else:
                    # if specified as a 2D tensor, feed Nth row to Nth neuron element (current format: units X steps).
                    # if current rows < num_units, the missing inputs will be 0s.
                    for j, row in enumerate(temp):
                        currents[i][j, :] = design_input_current(row, steps, 1)

        # describe the network to the user.
        num_neurons = sum([net.graph.nodes[i]["reference"].num_units for i in net.graph.nodes])

        logging.info(
            f"[Simulator] '{net.identifier}' has {num_neurons} neurons grouped into {len(net.graph.nodes)} "
            f"ensembles, connected by {len(net.graph.edges)} synapse matrices."
        )
        logging.info(f"[Simulator] Simulating {steps} steps at a resolution of {DT} ms.")

        # simulate the network.
        with alive_bar(total=steps, force_tty=True) as bar:
            for i in range(steps):
                root_inputs = [currents[j][:, i] for j in range(len(net.roots))]
                net(root_inputs)
                bar()

        # optional tensorboard logging.
        if tensorboard:
            writer = TensorboardWriter(log_dir=tb_dir)
            with alive_bar(total=len(os.listdir(data_dir)), force_tty=True) as bar:
                logging.info("[Simulator] Writing tensorboard data for visualization.")
                for file in os.listdir(data_dir):
                    if not file.endswith(".h5"):
                        continue

                    # write what was asked.
                    tensorboard_settings = self.configuration.get("simulation", {}).get("tensorboard", 0)
                    if type(tensorboard_settings) is list:
                        for task in tensorboard_settings:
                            for attr, settings in task.items():
                                # read only requested key from file.
                                data = HDFData(path=os.path.join(data_dir, file), key=attr)

                                # write the loggable or configurable attribute corresponding to this task.
                                writer.write(data, attr, **settings)
                    bar()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-config",
        action="store",
        dest="config",
        metavar="FILE",
        required=True,
        help="Path to hybrid model-simulation configuration YAML.",
    )
    args = parser.parse_args()

    # run the simulation with the given configuration file.
    Simulator(configuration=args.config).run()
