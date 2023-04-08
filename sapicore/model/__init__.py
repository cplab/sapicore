"""Train and test neuromorphic models.

Models are networks endowed with the methods :meth:`fit`, :meth:`predict`, :meth:`similarity`, :meth:`load`,
and :meth:`save`. This class provides an ML-focused API for the training and usage of sapicore
:class:`~engine.network.Network` output for practical purposes.

"""
from typing import Callable

import dill
import os

import networkx as nx
import matplotlib.pyplot as plt
from alive_progress import alive_bar

import torch
from torch import Tensor
from sklearn.base import BaseEstimator

from sapicore.data import Data
from sapicore.engine.network import Network
from sapicore.utils.io import ensure_dir


class Model(BaseEstimator):
    """Model base class.

    Implements and extends the scikit-learn :class:`sklearn.base.BaseEstimator` interface.

    Note
    ----
    In a machine learning context, spiking networks can be used in diverse ways, e.g. as classifiers
    or as generative models. While :mod:`engine` is mostly about form (network architecture and information flow),
    :mod:`model` is all about function (how to fit the model to data and how to utilize it once trained).

    """

    def __init__(self, network: Network = None, **kwargs):
        super().__init__()
        self.network = network

    def fit(self, data: Tensor | list[Tensor], repetitions: int | list[int] | Tensor = 1):
        """Applies :meth:`engine.network.Network.forward` sequentially on a block of samples `data`,
        then turns off learning for the network.

        The training samples may be obtained, e.g., from a :class:`~data.sampling.CV` cross validator object.

        Parameters
        ----------
        data: Tensor or list of Tensor
            2D tensor(s) of data samples to be fed to the root ensembles of this object's `network`,
            formatted sample X feature.

        repetitions: int or list of int or Tensor
            How many times to repeat each sample before moving on to the next one.
            Simulates duration of exposure to a particular input. If a list or a tensor is provided,
            the i-th row in the batch is repeated `repetitions[i]` times.

        Warning
        -------
        :meth:`~model.Model.fit` does not return intermediate output. Users should register forward hooks to
        efficiently stream data to disk throughout the simulation (see :meth:`~engine.network.Network.add_data_hook`).

        """
        # wrap 2D tensor data in a list if need be, to make subsequent single- and multi-root operations uniform.
        if not isinstance(data, list):
            data = [data]

        num_samples = data[0].shape[0]
        with alive_bar(total=num_samples, force_tty=True) as bar:
            # iterate over samples.
            for i in range(num_samples):
                # repeat each sample for as many time steps as instructed.
                for _ in range(repetitions if isinstance(repetitions, int) else repetitions[i]):
                    self.network([data[j][i, :] for j in range(len(data))])
                bar()

        # learning is turned off after fitting by default.
        for synapse in self.network.get_synapses():
            synapse.set_learning(False)

    def predict(self, data: Data | Tensor) -> Tensor:
        """Predicts the labels of `data` by feeding the samples to a trained network and applying
        some procedure to the resulting population/readout layer response.

        Parameters
        ----------
        data: Data or Tensor
            Sapicore dataset or a standalone 2D tensor of data samples, formatted sample X feature.

        Returns
        -------
        Tensor
            Vector (1D tensor) of predicted labels.

        """
        pass

    def similarity(self, data: Tensor, metric: str | Callable) -> Tensor:
        """Performs rudimentary similarity analysis on the network population responses to `data`,
        obtaining a pairwise distance matrix reflecting sample separation.

        Parameters
        ----------
        data: Tensor
            2D tensor of data samples, formatted sample X feature.

        metric: str or Callable
            Distance metric to be used. If a string value is provided, it should correspond to one of the available
            :mod:`scipy.spatial.distance` metrics. If a custom function is provided, it should accept the
            `data` tensor and return a scalar corresponding to their distance.

        """
        pass

    def save(self, path: str):
        """Saves the :class:`engine.network.Network` object owned by this model to `path`.

        Parameters
        ----------
        path: str
            Destination path, inclusive of the file to which the network should be saved.

        Note
        ----
        The default implementation uses :meth:`torch.save`, for the common case where files are used.
        Users may override this method when other formats are called for.

        """
        ensure_dir(os.path.dirname(path))
        torch.save(self.network, path, pickle_module=dill)

    def load(self, path: str) -> Network:
        """Loads a :class:`engine.network.Network` from `path` and assigns it to this object's `network` attribute.

        Parameters
        ----------
        path: str
            Path to the file containing the model.

        Returns
        -------
        Network
            A reference to the loaded :class:`engine.network.Network` object in case it is
            required by the calling function.

        Note
        ----
        The default implementation uses :meth:`torch.load`, for the common case where files are used.
        Users may override this method when other formats are called for.

        """
        if os.path.exists(path):
            self.network = torch.load(path, pickle_module=dill)

        return self.network

    def draw(self, path: str, node_size: int = 750):
        """Saves an SVG networkx graph plot showing ensembles and their general connectivity patterns.

        Parameters
        ----------
        path: str
            Destination path for network figure.

        node_size: int, optional
            Node size in network graph plot.

        Note
        ----
        May be extended and/or moved to a dedicated visualization package in future versions.

        """
        plt.figure()
        nx.draw(
            self.network.graph, node_size=node_size, with_labels=True, pos=nx.kamada_kawai_layout(self.network.graph)
        )

        plt.savefig(fname=os.path.join(path, self.network.identifier + ".svg"))
        plt.clf()
