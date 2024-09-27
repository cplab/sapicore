"""Train and test neuromorphic models.

Models are networks endowed with the methods :meth:`fit`, :meth:`predict`, :meth:`similarity`, :meth:`load`,
and :meth:`save`. This class provides an ML-focused API for the training and usage of sapicore
:class:`~engine.network.Network` output for practical purposes.

"""
from typing import Callable, Sequence

import dill
import os

from alive_progress import alive_bar

import torch
from torch import Tensor

from sapicore.engine.network import Network
from sapicore.utils.io import ensure_dir


class Model:
    """Model base class.

    Loosely follows the design of scikit-learn's :class:`sklearn.base.BaseEstimator` interface.

    Note
    ----
    In a machine learning context, spiking networks can be used in diverse ways, e.g. as classifiers
    or as generative models. While :mod:`engine` is mostly about form (network architecture and information flow),
    :mod:`model` is all about function (how to fit the model to data and how to utilize it once trained).

    """

    def __init__(self, network: Network = None, **kwargs):
        # store a reference to one network object.
        self.network = network

    def serve(
        self, data: Tensor | Sequence[Tensor], duration: int | Sequence[int], rinse: int | Sequence[int] = 0, **kwargs
    ):
        """Applies :meth:`engine.network.Network.forward` sequentially on a batch of buffer `data`.

        Parameters
        ----------
        data: Tensor or Sequence of Tensor
            2D tensor(s) of data buffer to be fed to the root ensemble(s) of this object's `network`,
            formatted sample X feature.

        duration: int or Sequence of int
            Duration of sample presentation. Simulates duration of exposure to a particular input.
            If a list or a tensor is provided, the i-th sample in the batch is maintained for `duration[i]` steps.

        rinse: int or Sequence of int
            Null stimulation steps (0s in-between samples).
            If a list or a tensor is provided, the i-th sample is followed by `rinse[i]` rinse steps.
        Each sample `i` is presented for `duration[i]`, followed by all 0s stimulation for `rinse[i]`.

        """
        # wrap 2D tensor data in a list if need be, to make subsequent single- and multi-root operations uniform.
        if not isinstance(data, Sequence):
            data = [data]

        num_samples = data[0].shape[0]

        with alive_bar(total=num_samples, force_tty=True) as bar:
            for i in range(num_samples):
                # compute presentation and rinse durations.
                stim_steps = duration if isinstance(duration, int) else duration[i]
                rinse_steps = rinse if isinstance(rinse, int) else rinse[i]

                for k in range(stim_steps):
                    # stimulus presentation.
                    self.network.forward([data[j][i, :] for j in range(len(data))])

                for _ in range(rinse_steps):
                    # inter-stimulus interval.
                    self.network.forward([torch.zeros_like(data[j][i, :]) for j in range(len(data))])

                # advance progress bar.
                bar()

    def fit(
        self, data: Tensor | Sequence[Tensor], duration: int | Sequence[int], rinse: int | Sequence[int] = 0, **kwargs
    ):
        """Serves a batch of data, then turns off learning for all synapses.

        Warning
        -------
        :meth:`~model.Model.fit` does not return intermediate output. Users should register forward hooks to
        efficiently stream data to memory or disk throughout the simulation
        (see :meth:`~engine.network.Network.add_data_hook`).

        """
        self.serve(data, rinse, duration)

        # after fitting a model, learning is turned off for all synapses by default.
        for synapse in self.network.get_synapses():
            synapse.set_learning(False)

    def predict(self, data: Tensor, labels: Sequence, **kwargs) -> Sequence:
        """Predicts the labels of `data`.

        Parameters
        ----------
        data: Tensor
            Standalone 2D tensor of data buffer, sample X feature.

        labels: Sequence
            Label values corresponding to classification layer cell indices.

        Returns
        -------
        Sequence
            Vector of predicted labels.

        """
        raise NotImplementedError

    def similarity(self, data: Tensor, metric: str | Callable, **kwargs) -> Tensor:
        """Performs a similarity analysis on network responses to `data`, yielding a pairwise distance matrix.

        Parameters
        ----------
        data: Tensor
            2D tensor of data buffer, formatted sample X feature.

        metric: str or Callable
            Distance metric to be used. If a string value is provided, it should correspond to one of the available
            :mod:`scipy.spatial.distance` metrics. If a custom function is provided, it should accept the
            `data` tensor and return a scalar corresponding to their distance.

        """
        raise NotImplementedError

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
