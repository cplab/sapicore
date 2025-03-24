""" File and user I/O. """
from typing import Callable, Sequence

import _io

import os
import sys
import logging

from pathlib import Path
from itertools import chain

import yaml
import h5py as hdf

import torch
from torch import Tensor
from torch.nn import Module

from alive_progress import alive_bar
from tree_config import apply_config, load_config

from sapicore import __version__
from sapicore.utils.tensorboard import TensorboardWriter, HDFData

__all__ = (
    "DataAccumulatorHook",
    "MonitorHook",
    "ensure_dir",
    "flatten",
    "load_yaml",
    "save_yaml",
    "log_settings",
    "load_apply_config",
    "plot_tensorboard",
)

CHUNK_SIZE = 50.0
""" Data chunks will be dumped to disk after reaching this size in MB. """


class MonitorHook(Module):
    """Accumulates selected attribute data in memory on every forward call. Does not write to disk.

    Parameters
    ----------
    component: torch.nn.Module
        Network component whose attributes should be buffered in memory.

    attributes: list of str
        Instance attributes to be logged. Defaults to the full list `component.loggable_props`.

    entries: int
        Expected number of simulation steps. Used to preallocate the buffer.
        If unknown, `torch.vstack()` is used, which may result in slower performance.

    dtype:
        Data type.

    """

    def __init__(self, component: torch.nn.Module, attributes: Sequence[str], entries: int = None, dtype=torch.float):
        super().__init__()

        self.component = component
        self.attributes = component.loggable_props if not attributes else attributes

        self.iteration = 0
        self.entries = entries

        self.cache = {}
        self.dtype = dtype

        # (de)activate hook without removing it.
        self.active = True
        self.handle = component.register_forward_hook(self.monitor_hook())

    def __getitem__(self, item):
        if item is not None and not isinstance(item, slice):
            return self.cache[item] if item in self.cache.keys() else {}
        else:
            # [:] will be a shorthand for returning the cache dictionary.
            return self.cache

    def remove(self):
        self.handle.remove()

    def pause(self):
        self.active = False

    def unpause(self):
        self.active = True

    def monitor_hook(self) -> Callable:
        def fn(_, __, output):
            if self.active:
                # add current outputs to this data accumulator instance's cache dictionary, whose values are tensors.
                for attr in self.attributes:
                    odim = output[attr].dim()
                    if attr not in self.cache.keys():
                        # the cache dictionary is empty because this is the first iteration.
                        if self.entries is None:
                            # expand output by one dimension (zero axis) to fit.
                            self.cache[attr] = output[attr][None, :] if odim == 1 else output[attr][None, :, :]
                        else:
                            # preallocate if number of steps is known.
                            dim = [self.entries, len(output[attr])] + ([output[attr].shape[1]] if odim == 2 else [])
                            self.cache[attr] = torch.empty(dim, device=self.component.device, dtype=self.dtype)

                    else:
                        if self.entries is None:
                            # vertically stack output attribute to cache tensor at the appropriate key.
                            target = output[attr][None, :] if odim == 1 else output[attr][None, :, :]
                            self.cache[attr] = torch.vstack([self.cache[attr], target])
                        else:
                            # update appropriate row in preallocated tensor.
                            self.cache[attr][self.iteration] = output[attr]

                # advance iteration counter.
                self.iteration += 1

        return fn


class DataAccumulatorHook(Module):
    """Wraps any :class:`~torch.nn.Module` object, accumulating selected attribute data on every forward call.

    If used with a :class:`~engine.component.Component`, selectable attributes are those included in the dictionary
    returned by the Module's forward method, and are a subset of the object's `_loggable_props_`.

    Parameters
    ----------
    component: torch.nn.Module
        Reference to the network component object whose loggable buffered attributes should be dumped to disk.

    log_dir: str
        Path to the desired data loggable directory.

    attributes: list of str
        Instance attribute names to be logged. Defaults to the full list `component.loggable_props`.

    entries: int
        Number of log entries, equal to the number of simulation steps. Used to ensure the last batch in the run
        is written to disk despite its size being smaller than :attr:`CHUNK_SIZE`.

    """

    def __init__(self, component: torch.nn.Module, log_dir: str, attributes: Sequence[str], entries: int):
        super().__init__()

        self.log_dir = log_dir
        self.component = component
        self.attributes = component.loggable_props if not attributes else attributes
        self.hdf_file_path = os.path.join(self.log_dir, self.component.identifier + ".h5")

        # total number of forward calls to expect in this simulation run (steps).
        self.entries = entries

        # tensor for caching input until memory threshold is reached, at which point a chunk will be saved to disk.
        self.cache = {}
        self.chunk_size = {}

        # counter for iterations, to be checked against `entries` to ensure all data is written to disk.
        self.iterations = 0

        # initialize HDF5 files and register the forward hook.
        self.initialize_hdf()
        component.register_forward_hook(self.save_outputs_hook())

    def initialize_hdf(self) -> None:
        """Initializes HDF5 file on the first iteration of this data accumulator instance."""
        # detects whether we are in the first iteration before creating or appending to the file.
        if not os.path.exists(self.hdf_file_path):
            ensure_dir(os.path.dirname(self.hdf_file_path))
            with hdf.File(os.path.join(self.log_dir, self.component.identifier + ".h5"), "a") as hf:
                # write metadata to be able to identify this HDF with its object.
                hf.attrs["identifier"] = self.component.identifier
                hf.attrs["class"] = type(self.component).__name__
                hf.attrs["version"] = __version__

                # create datasets for loggable properties.
                # max shape unlimited in first axis, so that it can be expanded on every simulation step.
                for attr in self.attributes:
                    attr_tensor = getattr(self.component, attr)
                    expanded_dimensions = attr_tensor[None, :].shape
                    self.chunk_size[attr] = 0

                    hf.create_dataset(
                        name=attr,
                        shape=expanded_dimensions,
                        compression="lzf",
                        chunks=True,
                        maxshape=tuple([s if j > 0 else None for j, s in enumerate(expanded_dimensions)]),
                    )

                # write configurable properties ONCE for future reference, e.g. during tensorboard loggable.
                if hasattr(self.component, "_config_props"):
                    hf.create_group(name="configuration")
                    for prop in getattr(self.component, "_config_props"):
                        # log configurable attribute values as metadata in the "configuration" group.
                        value = getattr(self.component, prop)
                        hf["/configuration"].attrs[prop] = repr(value)
        else:
            raise FileExistsError("Target HDF5 log file already exists.")

    def save_outputs_hook(self) -> Callable:
        def fn(_, __, output):
            """Updates the output cache and appends to destination HDF5 file if cache is full or simulation ends."""
            # increment iteration counter.
            self.iterations += 1

            # add current outputs to this data accumulator instance's cache dictionary, whose values are tensors.
            for attr in self.attributes:
                if attr not in self.cache.keys():
                    # the cache dictionary is empty because this is the first iteration post-dump or in general.
                    self.cache[attr] = output[attr][None, :]  # expand output by one dimension (zero axis) to fit.
                else:
                    # vertically stack output attribute to cache tensor at the appropriate key.
                    self.cache[attr] = torch.vstack([self.cache[attr], output[attr][None, :]])

            # calculate cache size (MB).
            cache_size = sum([v.element_size() * v.nelement() for _, v in self.cache.items()]) / 1000000.0

            # if memory usage exceeds the limit given or reached the end of the run, append cached content to HDF5.
            if cache_size > CHUNK_SIZE or self.iterations >= self.entries:
                with hdf.File(os.path.join(self.log_dir, self.component.identifier + ".h5"), "a") as hf:
                    for attr in self.attributes:
                        # increase data chunk size by number of rows in the cache tensor.
                        self.chunk_size[attr] += self.cache[attr].shape[0]
                        hf[attr].resize(self.chunk_size[attr], axis=0)

                        # write current cache tensor to its respective HDF5 data structure at the correct indices.
                        start_index = hf[attr].shape[0] - self.cache[attr].shape[0]
                        end_index = hf[attr].shape[0]

                        # detach().numpy() necessary if the tensor requires grad.
                        hf[attr][start_index:end_index, :] = self.cache[attr].cpu().detach().numpy()

                    # clear cache.
                    self.cache = {}

        return fn

    def forward(self, data: Tensor) -> None:
        _ = self.component(data)
        return


def log_settings(
    level: int = logging.INFO,
    stream: _io.TextIOWrapper = sys.stdout,
    file: str = None,
    formatting: str = "%(asctime)s [%(levelname)s] %(message)s",
):
    """Temporary basic logger configuration.

    Parameters
    ----------
    level: int
        Logger level, `INFO` by default.

    stream:
        Destination stream, `stdout` by default.

    file:
        Destination file, `None` by default.

    formatting: str
        Message formatting.

    """
    logging.basicConfig(
        level=level,
        format=formatting,
        handlers=[logging.StreamHandler(stream)] + ([logging.FileHandler(file)] if file else []),
    )


def load_apply_config(path: str, apply_to: object = None) -> dict:
    # parse configuration from YAML if given and use resulting dictionary to initialize attributes.
    content = load_config(None, path) if os.path.exists(path) else None

    if not content:
        raise FileNotFoundError(f"Could not find a configuration YAML at {os.path.realpath(path)}")

    else:
        # apply the configuration to this pipeline object.
        configuration = content

        if apply_to:
            apply_config(apply_to, configuration)

    return configuration


def ensure_dir(path: str = None) -> str:
    """Ensures that a path is created if parts thereof do not exist yet and returns it.

    Parameters
    ----------
    path: str
        File path to be verified or created.

    Returns
    -------
    path: str
        The same file path, so it can be used in the calling context (e.g., if :func:`ensure_dir`
        wraps an assignment).

    """
    os.makedirs(path, exist_ok=True)
    return path


def flatten(lst: list[list]) -> list:
    """Flattens a list."""
    return list(chain(*lst))


def load_yaml(path: str) -> dict:
    """Parses the YAML at `path`, returning a dictionary."""
    return yaml.safe_load(Path(path).read_text())


def save_yaml(data: dict, path: str):
    """Saves a dictionary `data` to the YAML file specified by `path`."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def plot_tensorboard(configuration: dict, data_dir: str, tb_dir: str):
    """Loads simulation output data from disk and writes it to tensorboard for visual inspection."""
    # REFACTOR when rewriting this more formally, need a way to pass arbitrary plot settings that can't
    #  necessarily go through YAML, e.g. experiment event timestamps and label colors.
    writer = TensorboardWriter(log_dir=tb_dir)
    with alive_bar(total=len(os.listdir(data_dir)), force_tty=True) as bar:
        logging.info("Writing tensorboard data for visualization.")
        for file in os.listdir(data_dir):
            if not file.endswith(".h5"):
                continue

            # write what was asked.
            tensorboard_settings = configuration.get("simulation", {}).get("tensorboard", 0)
            if isinstance(tensorboard_settings, list):
                for task in tensorboard_settings:
                    for attr, settings in task.items():
                        # read only requested key from file.
                        data = HDFData(path=os.path.join(data_dir, file), key=attr)

                        # write the loggable or configurable attribute corresponding to this task.
                        writer.write(data, attr, **settings)
            bar()
