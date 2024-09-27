""" Data operations. """
from typing import Any, Callable, Optional, Sequence
from numpy.typing import NDArray

import os
import urllib.request
from urllib.error import HTTPError, URLError

from copy import deepcopy
from glob import glob
from natsort import natsorted

import torch
import numpy as np
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset

from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import normalize as norm

from sapicore.data.sampling import BalancedSampler, CV
from sapicore.utils.io import ensure_dir


__all__ = ("AxisDescriptor", "Metadata", "Data")


class AxisDescriptor:
    """Metadata vector of `labels` describing a particular axis of an N-dimensional array.

    Parameters
    ----------
    name: str, optional
        Name of the descriptor (i.e., the variable whose value is given by the labels).

    labels: list or NDarray or Tensor, optional
        Label values given as a list, numpy array, or tensor. Regardless of passed type, these are internally
        converted to a numpy array (not tensors, as those do not support string labels).

    axis: int, optional
        The data axis described by this variable. Defaults to 1 (columns).

    Example
    -------
    Generate a dummy dataset and describe its axes with multiple lists of labels:

        >>> import torch
        >>> data = Data(buffer=torch.rand(8, 4))
        >>> study = AxisDescriptor(name="study", labels=[1, 1, 2, 2, 1, 1, 2, 2], axis=0)
        >>> animal = AxisDescriptor(name="animal", labels=["A", "A", "A", "A", "B", "B", "B", "B"], axis=0)
        >>> sensor = AxisDescriptor(name="sensor", labels=["Occipital", "Parietal", "Temporal", "Frontal"], axis=1)

    Here, we have eight buffer (rows) of four measured dimensions (columns). The third AxisDescriptor indicates
    that the columns correspond to sensor locations. The first two describe the rows, and contain information
    about the study and animal each sample was obtained from.

    """

    def __init__(self, name: str = "", labels: list | NDArray | Tensor = None, axis: int = 0):
        self.name = name
        self.axis = axis
        self.labels = np.array(labels)

    def __getitem__(self, index: Any) -> Tensor:
        return self.labels[index]

    def __setitem__(self, index: Any, values: Any):
        self.labels[index] = np.array(values)


class Metadata:
    """Collection of :class:`AxisDescriptor` objects indexed in a table."""

    def __init__(self, *args: AxisDescriptor):
        # verify that no duplicate descriptors exist in args.
        self._validate_names(*args)

        # dictionary containing descriptor names as keys and their references as values.
        self.table = {descriptor.name: descriptor for descriptor in args}

    def __getitem__(self, key: str) -> AxisDescriptor:
        """Return the label vector `key` if exists."""
        if key in self.table.keys():
            return self.table[key]

        else:
            raise KeyError(f"Metadata does not contain the key {key}.")

    def __setitem__(self, key: str, value: list | NDArray | Tensor | AxisDescriptor):
        """Creates an :class:`AxisDescriptor` from the `value` vector and adds it to this instance as
        a row descriptor (axis=0).

        Note
        ----
        Guarantees that the `name` attribute of the AxisDescriptor matches its key in `table`.

        """
        if isinstance(value, AxisDescriptor):
            if value.name != key:
                raise ValueError(f"Key does not match ({key} != {value.name})")

            self.table[key] = value

        else:
            self.table[key] = AxisDescriptor(name=key, labels=np.array(value), axis=0)

    def add_descriptors(self, *args: AxisDescriptor):
        """Adds one or more :class:`AxisDescriptor` objects to this metadata instance."""
        self._validate_names(*args)

        for descriptor in args:
            self.table[descriptor.name] = descriptor

    def to_dataframe(self, axis: int = 0) -> pd.DataFrame:
        """Condenses the :class:`~data.Data.AxisDescriptor` objects registered in this dataset's
        :class:`data.Data.Metadata` `table` attribute to a pandas dataframe.

        Performed for row (sample) descriptors by default, but can also be used to aggregate
        :class:`AxisDescriptor` objects describing any other `axis`.

        Note
        ----
        If descriptors for a particular axis vary in length, NaN values will be filled in as necessary.

        """
        df = pd.DataFrame()
        for key, values in self.table.items():
            if values.axis == axis:
                df[key] = pd.Series(values.labels)

        return df

    @staticmethod
    def _validate_names(*args: AxisDescriptor):
        if len(args) != len(set(descriptor.name for descriptor in args)):
            raise ValueError(f"Got multiple descriptors with the same name in {args}")


class Data(Dataset):
    """Dataset base class.

    Provides an interface and basic default implementations for fetching, organizing,
    and representing external datasets. Designed to be incrementally extended.

    Parameters
    ----------
    identifier: str, optional
        Name for this dataset.

    metadata: Metadata, optional
        Metadata object describing this dataset. Consists of multiple :class:`AxisDescriptor` references.

    root: str, optional
        Local root directory for dataset file(s), if applicable.

    remote_urls: str or list of str, optional
        Remote URL(s) from which to fetch this dataset, if applicable.

    buffer: Tensor, optional
        A buffer holding data buffer in tensor form. Useful for initializing objects on the fly, e.g.
        during data synthesis. Users may leverage :meth:`access` to manage disk I/O, e.g.
        using :class:`torch.storage.Storage`, memory mapped arrays/tensors, or HDF lazy loading.

    download: bool, optional
        Whether to download the set. Defaults to `False`.
        If `True`, the download only commences if the `root` doesn't exist or is empty.

    overwrite: bool, optional
        Whether to download the set regardless of whether `root` already contains cached/pre-downloaded data.

    labels: Sequence, optional
        Labels to create row metadata from, using a default procedure.

    See Also
    --------
    `PyTorch Storage Class <https://discuss.pytorch.org/t/memory-mapped-tensor/8954>`_

    """

    def __init__(
        self,
        identifier: str = "",
        buffer: Tensor = None,
        metadata: Metadata | Sequence = None,
        root: Optional[str] = None,
        remote_urls: str | list[str] = "",
        download: bool = False,
        overwrite: bool = False,
        labels: Sequence = None,
        **kwargs,
    ):
        # dataset identifier, sometimes used with default methods.
        self.identifier = identifier

        # optional data tensor provided by user at instantiation.
        self.buffer = buffer

        # metadata object containing axis-specific descriptor labels.
        if isinstance(metadata, Metadata):
            self.metadata = metadata
        else:
            self.metadata = Metadata()
            if isinstance(metadata, Sequence) and labels is not None:
                # if labels supplied, use them to create row metadata.
                self.metadata.add_descriptors(AxisDescriptor(labels))

        # optional, local and remote directories.
        self.root = root
        self.remote_urls = remote_urls

        # fetch dataset from remote URL if `root` doesn't exist or is empty or if asked to do so regardless.
        if overwrite or (download and not self._check_downloaded()):
            self._download()

            # converts the foreign dataset to a single file in a user-determined format.
            # passes silently if not implemented by the user.
            self._standardize()

    def __getitem__(self, index: Any):
        """Calls :meth:`access` to slice into the data or access specific file(s), returning the value(s) at `index`."""
        return self.access(index)

    def __setitem__(self, index: Any, values: Tensor):
        """Sets buffer values at the given indices to `values`."""
        self.modify(index, values)

    def __len__(self):
        """The default implementation addresses trivial cases where the set is loaded to memory (in `self.buffer`).

        Warning
        -------
        Users should override this method when dealing with large datasets that are dynamically loaded using
        a :meth:`access` call.

        """
        return len(self.buffer)

    def _check_downloaded(self) -> bool:
        """Checks whether files have already been downloaded.

        By default, returns `True` if root directory exists and is nonempty.

        Note
        ----
        Users may wish to override this behavior in their own derivative dataset-specific classes,
        e.g. to include checksum verification.

        """
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def _download(self):
        """Downloads one or more files to the `root` directory from `remote_urls`.

        Raises
        ------
        TypeError
            If a URL is not provided and the data does not exist locally at `root`.

        ValueError
            If the URL provided is invalid.

        ConnectionError
            If remote URL is down.

        """
        # create the destination directory if it doesn't exist.
        ensure_dir(self.root)

        # if given a single URL, wrap it in a list.
        if isinstance(self.remote_urls, str):
            self.remote_urls = [self.remote_urls]

        # download file(s).
        for url in self.remote_urls:
            try:
                file = url.split("/")[-1]
                urllib.request.urlretrieve(url, os.path.join(self.root, file))
            except HTTPError as e:
                raise ConnectionError(f"Download failed, server down: {url}") from e
            except URLError as e:
                raise ConnectionError(f"Download failed, URL invalid: {url}") from e

    def _standardize(self):
        """Standardizes an external data directory tree, e.g. a remote repository, possibly converting it to a
        single condensed file at `root` for subsequent operations. Should only be performed once after fetching
        the data from its remote source, if applicable.

        This method is meant to be implemented by child classes that deal with arbitrarily formatted external sets.
        It may also serve as a bridge between Sapicore and specialized libraries, e.g. MNE for EEG/MEG data.

        Warning
        -------
        The resulting file may follow any convention the user would like to adopt. Sapicore makes no assumptions;
        it is the user's responsibility to implement this method and :meth:`access` below in a compatible way.

        """
        pass

    def get_metadata(self, keys: bool = True) -> list[Any]:
        """Returns metadata keys by default, or their values (AxisDescriptor references) if `keys` is False."""
        return list(self.metadata.table.keys()) if keys else list(self.metadata.table.values())

    def access(self, index: Any, axis: int = None) -> Tensor:
        """Specifies how to access data by mapping indices to actual samples (e.g., from file(s) in `root`).

        The default implementation slices into `self.buffer` to accommodate the trivial cases where the user has
        directly initialized this :class:`Data` object with a `buffer` tensor or loaded its values by reading
        a file that fits in memory (the latter case would be handled by :meth:`load`).

        More sophisticated use cases may require lazy loading or navigating HDF files. That kind of logic should
        be implemented here by derivative classes.

        Parameters
        ----------
        index: Any
            Index(es) to slice into.

        axis: int, optional
            Optionally, a specific axis along which to apply index selection.

        Note
        ----
        Where file data are concerned (e.g., audio/image, each being a labeled "sample"), use this method to read
        and potentially transform them, returning the finished product.

        """
        return self.buffer.index_select(axis, torch.as_tensor(index)) if axis is not None else self.buffer[index]

    def load(self, indices: Any = None):
        """Populates the `buffer` tensor buffer and/or `descriptors` attribute table by loading one or more files
        into memory, potentially selecting only `indices`.

        Since different datasets and pipelines call for different formats, implementation is left to the user.

        Parameters
        ----------
        indices: Any
            Specific indices to include, one for each file.

        Returns
        -------
        Data
            Self reference. For use in single-line initialization and loading.

        Warning
        -------
        Populating the `buffer` with the entire dataset should only be done when it can fit in memory.
        For large sets, the buffer should not be used naively; :meth:`access` should be overriden
        to implement some form of lazy loading.

        """
        pass

    def modify(self, index: Any, values: Tensor):
        """Set or modify data values at the given indices to `values`.

        The default implementation edits the `buffer` field of this :class:`Data` object.
        Users may wish to override it in cases where the buffer is not used directly.

        Parameters
        ----------
        index: Any
            Indices to modify.

        values: Tensor
            Values to set data at indices to.

        """
        self.buffer[index] = values

    def normalize(self, p: int = 1, axis: int = 1) -> Tensor:
        """
        Apply Lp normalization along a specified axis to the buffer.

        Parameters
        ----------
        p: int, optional
            The order of the norm (default is 1, which corresponds to L1 normalization).

        axis: int, optional
            The axis along which to normalize (default is 0).

        Returns
        -------
        Tensor
            The normalized buffer.

        """
        self.buffer = torch.as_tensor(norm(self.buffer.numpy(), axis=axis, norm=f"l{p}"))
        return self.buffer

    def save(self):
        """Dump the buffer contents and metadata to disk at `root`.

        Since different datasets and pipelines call for different formats, implementation is left to the user.

        """
        pass

    def scan(self, pattern: str = "*", recursive: bool = False) -> list[str]:
        """Scans the `root` directory and returns a list of files found that match the glob `pattern`.

        Parameters
        ----------
        pattern: str, optional
            Pattern to look for in file names that should be included, glob-formatted. Defaults to "*" (any).

        recursive: bool, optional
            Whether glob should search recursively. Defaults to `False`.

        """
        files = []
        if os.path.exists(self.root):
            for f in glob(os.path.join(self.root, pattern), recursive=recursive):
                files.append(f)

        return natsorted(files)

    def sample(self, method: Callable, axis: int = 0, **kwargs):
        """Applies `method` to sample from this dataset once without mutating it, returning
        a copy of the object containing only the data and labels at the sampled indices.

        The method can be a reference to a :class:`~sklearn.model_selection.BaseCrossValidator`.
        In that case, the keyword arguments should include any applicable keyword arguments,
        e.g. `shuffle`, `label_key`, `group_key` if applicable (see also :class:`~data.sampling.CV`).

        If `method` is not a base cross validator, keyword arguments will be passed to it directly.

        Parameters
        ----------
        method: Callable or BaseCrossValidator
            Method used to sample from this dataset.

        axis: int, optional
            Axis along which selection is performed. Defaults to zero (that is, rows/buffer).

        kwargs:
            retain: int or float
                The number or proportion of buffer to retain.

            shuffle: bool
                If using a :class:`sklearn.model_selection.BaseCrossValidator` to sample,
                whether to toggle the `shuffle` parameter on.

            label_keys: str or list of str
                Label key(s) by which to stratify sampling, if applicable.

            group_key: str or list of str
                Label key by which to group sampling, if applicable.

        Returns
        -------
        Data
            A subset of this dataset.

        """
        # determine whether a BaseCrossValidator was provided and if so, set it up.
        # `method` must be called because sklearn CV objects are ABCMeta (hence unrecognizable) until called.
        try:
            is_cv = isinstance(method(), BaseCrossValidator)
        except TypeError:
            is_cv = False

        if is_cv:
            # determine the number of folds based on `retain` and the dataset length as defined by __len__.
            # if `retain` not provided as a keyword argument, treat this as a 2-split.
            retain = kwargs.get("retain", 0.5)
            folds = int(self.__len__() * retain) if isinstance(retain, float) else self.__len__() // retain

            # if `shuffle` provided as a keyword argument, use its value in the method call (default to True).
            validator = method(folds, shuffle=kwargs.get("shuffle", True))
            cv_iterator = CV(data=self, cross_validator=validator, **kwargs)

            # use the first fold.
            _, subset = next(iter(cv_iterator))

        elif isinstance(method, BalancedSampler):
            # special case of the balanced sampler, which uses the aggregated descriptor dataframe.
            subset = method(frame=self.metadata.to_dataframe(axis), **kwargs)

        else:
            # in all other cases, directly apply the method provided.
            subset = method(**kwargs)

        # trim buffer and labels, returning a new partial dataset without mutating the original.
        return self.trim(index=subset, axis=axis)

    def trim(self, index: Any, axis: int = None):
        """Trims this instance by selecting `indices`, potentially along `axis`, returning a subset of the original
        dataset in terms of both buffer entries and labels/descriptors. Does not mutate the underlying object.

        Parameters
        ----------
        index: Any
            Index(es) to retain.

        axis: int, optional
            Single axis to apply index selection to. Defaults to None, in which case `index` is used directly
            within :meth:`access`.

        """
        # create a deep copy of this data object and trim its buffer and metadata.
        partial_data = deepcopy(self)
        partial_data.buffer = self.access(index, axis=axis)

        for k in partial_data.get_metadata():
            if partial_data.metadata[k].axis == axis or axis is None:
                partial_data.metadata[k] = partial_data.metadata[k][index]

        return partial_data

    def ingest(self, folds: int, shots: int, keys: str | Sequence[str], replace: bool = False):
        """Loads and samples the data, returning a balanced subset thereof.

        Parameters
        ----------
        folds: int
            Number of cross validation folds.

        shots: int
            Number of samples per class.

        keys: Sequence[str]
            Keys by which to group the data.

        replace: bool, optional
            Whether to sample with replacement. Defaults to `False`.

        """
        raise NotImplementedError
