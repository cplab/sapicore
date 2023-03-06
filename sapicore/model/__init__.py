"""Models provide an ML-focused API.

Models are networks endowed with fit() and predict() methods.

"""
from torch import Tensor
from sklearn.base import BaseEstimator

from sapicore.engine.network import Network


class Model(BaseEstimator):
    """Model base class.

    Implements the scikit-learn :class:`sklearn.base.BaseEstimator` interface.

    Note
    ----
    In a machine learning context, any spiking network can be used in multiple ways, e.g. as a classifier
    or as a generative model. While :mod:`engine` is mostly about form (network architecture and information flow),
    :mod:`model` is all about function (how to fit the model to data and how to use it once trained).

    """

    def __init__(self, network: Network, **kwargs):
        super().__init__()
        self.network = network

        # developer may override or define arbitrary attributes at instantiation.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, batch: Tensor, repetitions: int | list[int] | Tensor = 1):
        """Applies :meth:`engine.network.Network.forward` sequentially on a block of samples `batch`.

        The given samples may be obtained, e.g., from a :class:`data.sampling.CV` cross validator object.

        Parameters
        ----------
        batch: Tensor
            Data samples to be fed to `network`.

        repetitions: int or list of int or Tensor
            How many times to repeat each sample before moving on to the next one.
            Simulates duration of exposure to a particular input. If a list or a tensor is provided,
            the i-th row in the batch is repeated `repetitions[i]` times.

        """
        for i, row in enumerate(batch):
            [self.network(row) for _ in range(repetitions if isinstance(repetitions, int) else repetitions[i])]

    def predict(self, data: Tensor) -> Tensor:
        """Predicts the labels of samples in `data`.

        Specifies how to leverage this model class to perform classification.

        """
        pass
