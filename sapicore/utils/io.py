""" File and user I/O. """
import _io

import os
import sys
import logging

from pathlib import Path
from typing import Callable

import yaml
import h5py as hdf

import torch
from torch import Tensor
from torch.nn import Module

from tree_config import apply_config, load_config

__all__ = (
    "DataAccumulatorHook",
    "ensure_dir",
    "flatten",
    "load_yaml",
    "save_yaml",
    "log_settings",
    "load_apply_config",
)

CHUNK_SIZE = 50.0
""" Data chunks will be dumped to disk after reaching this size in MB. """


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

    def __init__(self, component: torch.nn.Module, log_dir: str, attributes: list[str], entries: int):
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
            with hdf.File(os.path.join(self.log_dir, self.component.identifier + ".h5"), "a") as hf:
                # write metadata to be able to identify this HDF with its object.
                hf.attrs["identifier"] = self.component.identifier
                hf.attrs["class"] = type(self.component).__name__

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
                    for prop in getattr(self.component, "_config_props"):
                        # create a dataset for this configurable property.
                        hf.create_dataset(name=prop, shape=attr_tensor.shape, compression="lzf")

                        # extract value to be logged and move to CPU if applicable.
                        value = getattr(self.component, prop)
                        if type(value) is Tensor:
                            # if tensor, move to cpu.
                            value = value.cpu()

                        try:
                            # store the configurable value or tensor in the dataset.
                            hf[prop][:] = value
                        except TypeError:
                            # skip field if not compatible with HDF5 (e.g., a method reference).
                            pass

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
            if cache_size > CHUNK_SIZE or self.iterations == self.entries:
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


def load_apply_config(configuration: str, apply_to: object = None) -> dict:
    # parse configuration from YAML if given and use resulting dictionary to initialize attributes.
    content = load_config(None, configuration) if os.path.exists(configuration) else None

    if not content:
        raise FileNotFoundError(f"Could not find a configuration YAML at {os.path.realpath(configuration)}")

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
    return [item for sublist in lst for item in sublist]


def load_yaml(path: str) -> dict:
    """Parses the YAML at `path`, returning a dictionary."""
    return yaml.safe_load(Path(path).read_text())


def save_yaml(data: dict, path: str):
    """Saves a dictionary `data` to the YAML file specified by `path`."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
