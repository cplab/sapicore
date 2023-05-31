""" Writes scalar and image data to tensorboard. """
import os

import numpy as np
import h5py as hdf

import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sapicore.utils.plotting import spike_raster


__all__ = ("HDFData", "TensorboardWriter")


class HDFData:
    """Reads HDF5 data logged to disk by :class:`~utils.io.DataAccumulatorHook`.

    Parameters
    ----------
    path: str, optional
        Path to HDF5 data file.

    key: str, optional
        Specific key to read from HDF5. If not provided, the entire file will be read.

    buffer: dict, optional
        For initializing an HDFData object with a preexisting dictionary, e.g. to save hyperparameters
        computed at runtime.

    Warning
    -------
    This class assumes that the HDF5 file only contains labeled datasets, not groups, in accordance with
    the current implementation of :class:`~utils.io.DataAccumulatorHook`.
    A generalized data loading protocol is being developed.

    """

    def __init__(self, path: str = None, key: str = None, buffer: dict = None):
        self.path = path
        self.key = key

        # buffer into which data stored in `keys` on files found in `path` will be loaded on request.
        self.buffer = buffer if buffer is not None else {}

        if self.path:
            self.read()

    def read(self):
        """Extracts desired datasets and/or fields from HDF5 file and adds them to the buffer."""
        #
        with hdf.File(os.path.realpath(self.path), "r") as f:
            # add component identifier and class for reference.
            try:
                self.buffer["identifier"] = f.attrs["identifier"]
                self.buffer["class"] = f.attrs["class"]

            except (AttributeError, ValueError):
                self.buffer["identifier"] = "NA"
                self.buffer["class"] = "NA"

            # add configurable property data to buffer.
            for attr in list(f.attrs):
                if not self.key or (self.key == attr):
                    self.buffer[attr] = f.attrs[attr]

            # add loggable property data to buffer.
            for attr in list(f):
                if not self.key or (self.key == attr):
                    self.buffer[attr] = f[attr][:]


class TensorboardWriter:
    """Reusable SummaryWriter instance endowed with methods handling tensorboard file I/O given a data object.

    For ease of use, :meth:`write` can be called with a multi-data buffer, e.g. in cases where
    the same plotting settings apply to all objects.

    Parameters
    ----------
    log_dir: str
        Path to target tensorboard directory.

    """

    def __init__(self, log_dir: str):
        """Initializes a `SummaryWriter` object."""
        self.writer = SummaryWriter(log_dir=log_dir)

    def write(self, data: HDFData, key: str, **settings):
        """Writes data to tensorboard using `settings`.

        Parameters
        ----------
        data: HDFData
            Object holding buffered data as a dictionary.

        key: str
            Key of configurable or loggable to process in this iteration.

        settings:
            The following optional arguments control how data is visualized:

            kind: str
                Kind of plot or data to write, currently supports "raster", "trace", "heatmap", "hparam".

            format: str
                Whether to write a 2D time series as an "image" or a "scalar" data to tensorboard.
                Will default to image for 3D time series (e.g., weight matrix + time).

            size: tuple of int
                Two-item tuple specifying figure size to be passed on to plt.rcParams["figure.figsize"].

            step: int
                Where applicable, size of skip step when plotting a time series on an interactive slider plot.
                E.g., we don't want to plot a weight matrix for every 1.0 ms simulation step, so we set step=10.

            time_axis: int
                Specify which axis of the array corresponds to time steps. Used for stacking images in
                interactive tensorboard slider plots.

        """
        # only process keys that exist in the file being handled.
        if key in data.buffer.keys():
            # extract component identifier from data buffer.
            name = data.buffer.get("identifier")

            if settings.get("kind") == "raster":
                # handle the raster plot case.
                self.raster(name=name, array=data.buffer[key], settings=settings)

            elif settings.get("kind") == "trace":
                # handle the individual trace case as either an image or a scalar.
                self.trace(name=name, attr=key, array=data.buffer[key], settings=settings)

            elif settings.get("kind") == "heatmap":
                # handle the heatmap case, use interactive slider plots if provided multiple.
                self.heatmap(name=name, attr=key, array=data.buffer[key], settings=settings)

            elif settings.get("kind") == "hparam":
                self.hparam(name=name, attr=key, array=data.buffer[key], settings=settings)

    def heatmap(self, name: str, attr: str, array: np.ndarray, settings: dict):
        """Adds 2D numeric data to tensorboard as a heatmap."""
        # handle case where user is trying to save a single heatmap as opposed to a time series thereof.
        if len(array.shape) != 3:
            # expand array along what would be the default time axis (rows).
            array = array[None, :]

        # extract number of steps from time_axis (or default to row dimension).
        num_steps = array.shape[settings.get("time_axis", 0)]

        # set an appropriate figure size (heatmaps are always images).
        plt.rcParams["figure.figsize"] = settings.get("size", (15, 10))
        plt.figure()

        for j in range(0, num_steps, settings.get("step", 100)):
            plt.grid(False)

            time_slice = array.take(indices=j, axis=settings.get("time_axis", 0))
            p = plt.imshow(time_slice, cmap=plt.cm.viridis, aspect="auto")

            limit = np.max(np.abs(array))
            plt.clim(-limit, limit)

            plt.colorbar()
            self.writer.add_figure(f"{name}/{attr}", p.get_figure(), global_step=j)

    def raster(self, name: str, array: np.ndarray | torch.Tensor, settings: dict):
        """Adds 1D integer binary data to tensorboard as a raster plot."""
        # set an appropriate figure size.
        plt.rcParams["figure.figsize"] = settings.get("size", (15, 10))

        # add a raster plot for the quantity (always an image).
        fig = plt.figure()
        ax = fig.add_subplot()

        collections = spike_raster(
            data=array,
            line_size=settings.get("line_size", 0.25),
            events=settings.get("events"),
            event_colors=settings.get("event_colors"),
        )

        if collections is not None:
            _ = [ax.add_collection(collection) for collection in collections]
            self.writer.add_figure(tag=f"{name}/raster", figure=fig)

    def trace(self, name: str, attr: str, array: np.ndarray, settings: dict):
        """Adds 1D numeric trace data to tensorboard as a scalar plot or an image."""
        time_axis = settings.get("time_axis", 0)
        data_axis = 1 if time_axis == 0 else 0

        for j in range(array.shape[data_axis]):
            if settings.get("format") == "scalar":
                # handle the scalar plot case
                for i in range(array.shape[time_axis]):
                    self.writer.add_scalars(f"{name}/N{j}", {f"{attr}": array[i, j]}, global_step=i)

            else:
                # set figure size and initialize empty figure.
                plt.rcParams["figure.figsize"] = settings.get("size", (10, 5))

                # generate a line plot for our scalar.
                fig = plt.figure()
                plt.plot(array[:, j])

                # set y-axis limits slightly beyond min and max values in data.
                height = (np.max(array) - np.min(array)) * 1.1
                if height > 0:
                    plt.ylim(np.mean(array) - height, np.mean(array) + height)

                # add figure to tensorboard.
                self.writer.add_figure(tag=f"{name}/N{j}/{attr}", figure=fig)

    def hparam(self, name: str, attr: str, array: np.ndarray, settings: dict):
        """Writes singleton hyperparameters, such as configurable attributes, to tensorboard.

        If you would like to visualize non-singleton configurable attributes (e.g., synaptic delays), please
        use one of the other plotting methods.

        Warning
        -------
        Tensorboard's `add_hparams()` will not allow writing different hparams sequentially (after the first one, they
        turn into scalars). It also cannot handle writing of a hyperparameter without an accompanying metric.

        Any workaround will likely require patching tensorboard, a task left to the intrepid maintainer.
        For now, configurable attributes can be logged as scalars for easy viewing of parameter value distributions.

        Sapicore may end up using a better maintained data visualization library.

        """
        # in the scalar case, iterate over elements being described (e.g., neurons in an ensemble), necessarily 1D.
        if settings.get("format") == "scalar":
            for i in range(array.shape[0]):
                self.writer.add_scalars(f"{name}/N{i}", {f"{attr}": array[i]})
