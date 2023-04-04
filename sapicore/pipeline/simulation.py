""" Simulate a network from a configuration YAML. """
import os
import logging

from argparse import ArgumentParser
from datetime import datetime

import torch
from torch import Tensor

from alive_progress import alive_bar

from sapicore.pipeline import Pipeline
from sapicore.model import Model
from sapicore.engine.network import Network

from sapicore.utils.seed import fix_random_seed
from sapicore.utils.signals import extend_input_current
from sapicore.utils.tensorboard import TensorboardWriter, HDFData

from sapicore.utils.constants import DT, TIME_FORMAT, SEED
from sapicore.utils.io import ensure_dir, save_yaml, log_settings

from sapicore.tests import ROOT

__all__ = ("Simulator",)


class Simulator(Pipeline):
    """A prototypical simulation pipeline, bringing together model specification, disk I/O, and visualization.
    Initialize a network and configure the simulation by providing a dictionary or a path to a YAML.

    Parameters
    ----------
    configuration: dict or str
        Configuration dictionary or path to a configuration YAML. If the path provided is relative, the simulation
        pipeline will assume that the root is `sapicore/tests` to support the execution of YAML-configured tests.

    """

    def __init__(self, configuration: dict | str, **kwargs):
        # loads the configuration if a path is provided.
        super().__init__(configuration=configuration, **kwargs)

        if self.configuration and self.configuration.get("device"):
            # set hardware device from configuration.
            self.device = self.configuration.get("device")

        else:
            # set default hardware device for this pipeline.
            self.device = "cpu" if not torch.cuda.is_available() else "cuda:0"

        self.initialize(**kwargs)

    def initialize(self, seed: int = None):
        """Initializes the simulation pipeline. Validates project root path, sets random number generation seed,
        and applies log message configuration.

        Parameters
        ----------
        seed: int, optional
            Random seed to be used throughout the simulation. Applied to all relevant libraries.
            If not provided, looks up "seed" in the configuration dictionary.
            If not found, the project default :attr:`utils.constants.SEED` is used to ensure reproducibility.

        Warning
        -------
        Deterministic behavior is vital when conducting reproducible experiments. This is one of many safeguards
        included in Sapicore. Users who wish to modify this default behavior will have to consciously override
        :meth:`~Simulator.initialize`.

        """
        # account for the relative path case, e.g. if invoked by pytest.
        project_root = self.configuration.get("root")
        if not project_root:
            raise KeyError("Configuration missing project 'root' key.")

        # account for the relative path case, e.g. if invoked by pytest.
        if not os.path.isabs(project_root):
            self.configuration["root"] = os.path.join(ROOT, project_root)

        # set random seed for entire simulation based on config, kwarg, or default (in this order of preference).
        random_seed = seed if seed else self.configuration.get("seed", SEED)
        fix_random_seed(random_seed)

        # apply default logger configuration.
        log_settings()

    def run(self):
        """Runs the simulation based on the given configuration.

        Raises
        ------
        KeyError
            If full path to project `root` directory is not provided as a key in the configuration YAML or dictionary.

        """
        # get full path to project root directory from configuration file.
        root = self.configuration.get("root")

        # create timestamped run directory.
        stamp = datetime.now().strftime(TIME_FORMAT)
        run_dir = ensure_dir(os.path.join(root, stamp))

        # save a copy of the configuration to this run's directory for reference.
        save_yaml(self.configuration, os.path.join(run_dir, self.configuration.get("identifier", "run_cfg") + ".yaml"))

        # load simulation settings from configuration dictionary.
        steps = self.configuration.get("simulation", {}).get("steps")
        disk = self.configuration.get("simulation", {}).get("disk")
        tensorboard = self.configuration.get("simulation", {}).get("tensorboard")

        # create data and tensorboard subdirectories.
        tb_dir = ensure_dir(os.path.join(run_dir, "tb"))
        data_dir = ensure_dir(os.path.join(run_dir, "data"))

        # initialize a model whose network has the given configuration.
        logging.info("Instantiating the network.")
        model = Model(network=Network(configuration=self.configuration, device=self.device))

        # describe the network to the user and plot its architecture.
        logging.info(model.network)
        model.draw(path=run_dir)

        # set up data accumulator hooks for streaming intermediate results to disk.
        # note that they are automatically applied on forward calls.
        if disk:
            model.network.data_hook(data_dir, steps)

        # if specified, design an input current time series for each root ensemble.
        currents = self._artificial_currents(model.network, steps)

        # run the simulation on the artificial currents.
        logging.info(f"Simulating {steps} steps at a resolution of {DT} ms.")
        model.fit(data=currents)

        # optional tensorboard logging.
        if tensorboard:
            self._tensorboard(data_dir, tb_dir)

    def _artificial_currents(self, net, steps) -> list[Tensor]:
        """Artificial input current generation.

        Warning
        -------
        Method will be extended and moved with the synthetic data update.

        """
        net_roots = [net.graph.nodes[i]["reference"] for i in net.roots]
        currents = [torch.zeros(size=(r.num_units, steps), dtype=torch.float, device=self.device) for r in net_roots]

        recipe = self.configuration.get("simulation", {}).get("current")
        if recipe:
            for i, root_node in enumerate(net.roots):
                # extract recipe for this root node.
                temp = torch.tensor(recipe.get(root_node))
                if temp.dim() == 1:
                    # if specified as a 1D tensor, feed same current to all neurons in the tensor.
                    currents[i] = extend_input_current(temp, steps, root_node.num_units)
                else:
                    # if specified as a 2D tensor, feed Nth row to Nth neuron element (current format: units X steps).
                    # if current rows < num_units, the missing inputs will be 0s.
                    for j, row in enumerate(temp):
                        currents[i][j, :] = extend_input_current(row, steps, 1)

        return [current.t() for current in currents]

    def _tensorboard(self, data_dir: str, tb_dir: str):
        """Loads simulation output data from disk and writes it to tensorboard for visual inspection.

        Warning
        -------
        Method will be extended and moved with the plotting update.

        """
        writer = TensorboardWriter(log_dir=tb_dir)
        with alive_bar(total=len(os.listdir(data_dir)), force_tty=True) as bar:
            logging.info("Writing tensorboard data for visualization.")
            for file in os.listdir(data_dir):
                if not file.endswith(".h5"):
                    continue

                # write what was asked.
                tensorboard_settings = self.configuration.get("simulation", {}).get("tensorboard", 0)
                if isinstance(tensorboard_settings, list):
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
