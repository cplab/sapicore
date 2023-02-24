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
from torch.nn import Module

__all__ = ("DataAccumulatorHook", "ensure_dir", "parse_yaml", "dump_yaml", "log_settings")

# data chunks will be dumped to disk after reaching this size in MB.
CHUNK_SIZE = 10.0


class DataAccumulatorHook(Module):
    """Wraps any :class:`~torch.nn.Module` object, accumulating selected attribute data on every forward call.
    Selectable attributes are those included in the dictionary returned by the Module's forward method.

    Parameters
    ----------
    component: torch.nn.Module
        Reference to the network component object whose loggable buffered attributes should be dumped to disk.

    log_dir: str
        Path to this run's data dump directory.

    attributes: list of str
        Instance attribute names to be logged. Defaults to the full list ``component.loggable_props`` if not specified.

    entries: int
        Number of log entries, equal to the number of simulation steps. Used to ensure the last batch in the run
        is written to disk despite its size being smaller than :attr:`~CHUNK_SIZE`.

    """

    def __init__(self, component: Module, log_dir: str, attributes: list[str], entries: int):
        super().__init__()

        self.log_dir = log_dir
        self.component = component
        self.attributes = component.loggable_props if not attributes else attributes
        self.hdf_file_path = os.path.join(self.log_dir, self.component.identifier + ".h5")

        # total number of forward calls to expect in this simulation run.
        self.entries = entries

        # tensor for caching input until memory is full, at which point the chunk will be dumped to disk.
        self.cache = {}
        self.chunk_size = {}
        # counter for iterations, to be checked against self.entries to ensure all data is written to disk.
        self.iterations = 0

        # initialize HDF5 files and register forward hook.
        self.initialize_state()
        component.register_forward_hook(self.save_outputs_hook())

    def initialize_state(self) -> None:
        """Initializes HDF5 file on the first iteration of this data accumulator instance."""
        # detects whether we are in the first iteration before creating or appending to the file.
        if not os.path.exists(self.hdf_file_path):
            with hdf.File(os.path.join(self.log_dir, self.component.identifier + ".h5"), "a") as hf:
                # write metadata to be able to identify this HDF with its nn.Module object.
                hf.attrs["identifier"] = self.component.identifier
                hf.attrs["class"] = type(self.component).__name__

                # write configurable properties for future reference and tensorboard logging.
                if hasattr(self.component, "_config_props_"):
                    for prop in getattr(self.component, "_config_props_"):
                        # extract value to be logged.
                        value = getattr(self.component, prop)

                        if type(value) is torch.Tensor:
                            # if tensor, move to cpu.
                            value = value.cpu()

                        try:
                            hf.attrs[prop] = value
                        except TypeError:
                            # skip field if not compatible with HDF5 (e.g., a method reference).
                            pass

                # create datasets for loggable properties. Max shape unlimited in first axis, used for the expansion.
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

    def save_outputs_hook(self) -> Callable:
        def fn(_, __, output):
            """Updates the output cache and appends to HDF5 file if memory is full or end is reached."""
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

    def forward(self, data: torch.tensor) -> None:
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
        Destination file, None by default.

    formatting: str
        Message formatting.

    Warning
    -------
    Future versions will include a Logger class and more extensive user-driven customization.

    """
    logging.basicConfig(
        level=level,
        format=formatting,
        handlers=[logging.StreamHandler(stream)] + ([logging.FileHandler(file)] if file else []),
    )


def ensure_dir(path: str = None) -> str:
    """Ensures that a path is created if parts thereof do not exist yet and returns it.

    Parameters
    ----------
    path: str
        File path to be verified or created.

    Returns
    -------
    path: str
        The same file path, so it can be used in the calling context (e.g., if :func:`ensure_dir` wraps an assignment).

    """
    os.makedirs(path, exist_ok=True)
    return path


def parse_yaml(path: str) -> dict:
    """Parses the YAML at `path`, returning a dictionary."""
    return yaml.safe_load(Path(path).read_text())


def dump_yaml(data: dict, path: str):
    """Saves a dictionary object `data` to the YAML file specified by `path`."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
