""" Simple simulation pipeline. """
import logging
import os

from argparse import ArgumentParser
from datetime import datetime

import matplotlib

import torch
from torch import Tensor

from sapicore.data import Data
from sapicore.model import Model
from sapicore.pipeline import Pipeline
from sapicore.engine.network import Network

from sapicore.utils.constants import DT, SEED, TIME_FORMAT, PLOT_BACKEND
from sapicore.utils.seed import fix_random_seed
from sapicore.utils.io import ensure_dir, log_settings, save_yaml, plot_tensorboard

__all__ = ("SimpleSimulator",)


class SimpleSimulator(Pipeline):
    """A prototypical simulation pipeline, bringing together data loading, network construction,
    model specification, disk I/O, and visualization.

    This pipeline can be configured by providing a dictionary or a path to a YAML.

    Parameters
    ----------
    configuration: dict or str
        Pipeline script configuration dictionary or path to YAML.

    data: Data or Tensor, optional
        Sapicore :class:`~data.Data` object or Tensor to use during model training.

    Note
    ----
    For a more complex demonstration involving data processing, selection, sampling, and cross-validation,
    see `tutorials/basic_experiment.py`. If using this script with data (be it Data or Tensor),
    it should be curated and coerced to the shape buffer X labels ahead of time.

    """

    def __init__(self, configuration: dict | str, data: Data | Tensor = None, **kwargs):
        super().__init__(config_or_path=configuration)
        matplotlib.use(PLOT_BACKEND)

        # set hardware device from configuration, default to GPU if available or CPU otherwise.
        self.device = self.configuration.get("device", "cpu" if not torch.cuda.is_available() else "cuda:0")

        # reference to data object containing buffer to be streamed to the network.
        self.data = data

        # project root directory, defaults to configuration YAML directory if given or empty otherwise.
        self.project_root = os.path.join(configuration, "..") if isinstance(configuration, str) else ""

        # prepare the environment by validating paths and setting a random seed for reproducibility.
        self.initialize(seed=kwargs.get("seed"))

    def initialize(self, seed: int = None):
        """Initializes the pipeline. Validates data object and project root path given in the configuration,
        sets random number generation seed, and applies log message configuration.

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
        :meth:`~SimpleSimulator.initialize`.

        """
        # update project root path from YAML configuration file if given.
        yaml_root = self.configuration.get("root")
        if yaml_root:
            if os.path.isabs(yaml_root):
                # if the given path is absolute, take it as is.
                self.project_root = yaml_root
            else:
                # otherwise, append it to the current value in project_root.
                # if the latter is empty, results in yaml_root relative to current working directory.
                self.project_root = os.path.join(self.project_root, yaml_root)

        # update full project root in configuration dictionary (used during network instantiation).
        self.project_root = os.path.realpath(self.project_root)
        self.configuration["root"] = self.project_root

        # prepare the dataset object by calling it (derived data classes implement __call__ appropriately).
        if self.data:
            if not (isinstance(self.data, Tensor) or isinstance(self.data, Data)):
                raise TypeError(f"Data object provided is of the unsupported type {type(self.data)}")

            if isinstance(self.data, Data):
                self.data = self.data.load()

            # moves data to the correct device (syntax is supported as it inherits from torch's Dataset).
            self.data = self.data[:].to(self.device)

        # set random seed for entire simulation based on config, kwarg, or default (in this order of preference).
        random_seed = seed if seed is not None else self.configuration.get("seed", SEED)
        fix_random_seed(random_seed)

        # apply default logger configuration.
        log_settings()

    def run(self):
        """Runs the generic simulator pipeline based on the given configuration."""
        # create timestamped directory for this simulation run in the project root.
        stamp = datetime.now().strftime(TIME_FORMAT)
        run_dir = ensure_dir(os.path.join(self.project_root, stamp))

        # load simulation settings from configuration dictionary.
        steps = self.configuration.get("simulation", {}).get("steps")
        disk = self.configuration.get("simulation", {}).get("disk")
        tensorboard = self.configuration.get("simulation", {}).get("tensorboard")

        # create data and tensorboard subdirectories.
        tb_dir = ensure_dir(os.path.join(run_dir, "tb"))
        data_dir = ensure_dir(os.path.join(run_dir, "data"))

        # save a copy of the master configuration file to this run's directory for future reference.
        save_yaml(self.configuration, os.path.join(run_dir, self.configuration.get("identifier", "config") + ".yaml"))

        # initialize a network using the given configuration and attach it to a model instance.
        logging.info("Instantiating the network.")
        model = Model(network=Network(configuration=self.configuration, device=self.device))

        # describe the network to the user.
        logging.info(model.network)

        if disk:
            # register data accumulator hooks for streaming intermediate output to disk (automatically applied).
            model.network.add_data_hook(data_dir, steps)

            # save an SVG plot of the network architecture.
            model.network.draw(path=run_dir)

        if not self.data:
            steps = self.configuration.get("simulation", {}).get("steps", {})
            if not steps:
                raise RuntimeError(
                    "Cannot run a simulation when neither data nor a number of simulation steps"
                    "(under the key simulation/steps in the configuration file) was given."
                )

            # if data not provided, set data to 0s of shape duration X root_node_num_units
            self.data = torch.zeros((steps, model.network[model.network.roots[0]].num_units), device=self.device)

            # if constant current value provided, add it to the dummy data tensor (useful for rudimentary testing).
            self.data = self.data + self.configuration.get("simulation", {}).get("current", 0.0)

        # fit the model by passing `data` buffer to the network (`duration` controls exposure time to each sample).
        logging.info(f"Simulating {steps} steps at a resolution of {DT} ms.")
        model.fit(data=self.data, duration=self.configuration.get("duration", 1))

        # optional tensorboard logging.
        if tensorboard:
            plot_tensorboard(self.configuration, data_dir, tb_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-config",
        action="store",
        dest="config",
        metavar="FILE",
        required=True,
        help="Path to simulator pipeline configuration file.",
    )
    args = parser.parse_args()

    # run the simulation with the given configuration file.
    SimpleSimulator(configuration=args.config).run()
