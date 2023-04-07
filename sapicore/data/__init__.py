""" Data operations. """
from typing import Callable

import os
import urllib.request

from copy import deepcopy
from glob import glob

import numpy as np
import pandas as pd
from natsort import natsorted

import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.model_selection import BaseCrossValidator

from sapicore.data.sampling import BalancedSampler, CV
from sapicore.utils.io import ensure_dir


__all__ = ("AxisDescriptor", "Data")


class AxisDescriptor:
    """Describes one axis of an N-dimensional :class:`Data` array with a vector of `labels`.

    Parameters
    ----------
    name: str, optional
        Name of the descriptor (i.e., the variable whose value is given by the labels).

    labels: list or Tensor, optional
        Label values given as a list.

    axis: int, optional
        The data axis described by this variable. Defaults to 1 (columns).

    Example
    -------
    Generate a dummy dataset and describe its axes with multiple lists of labels:

        >>> import torch
        >>> data = Data(samples=torch.rand(8, 4))
        >>> study = AxisDescriptor(name="study", labels=[1, 1, 2, 2, 1, 1, 2, 2], axis=0)
        >>> animal = AxisDescriptor(name="animal", labels=["A", "A", "A", "A", "B", "B", "B", "B"], axis=0)
        >>> sensor = AxisDescriptor(name="sensor", labels=["Occipital", "Parietal", "Temporal", "Frontal"], axis=1)

    Here, we have eight samples (rows) of four measured dimensions (columns). The third AxisDescriptor indicates
    that the columns correspond to EEG sensor locations. The first two describe the rows, and contain information
    about the study and animal each sample was obtained from.

    """

    def __init__(self, name: str = None, labels: list | Tensor = None, axis: int = 1):
        self.name = name
        self.axis = axis
        self.labels = labels

    def __getitem__(self, index: int | tuple[int]):
        return self.labels[index]

    def __str__(self):
        return self.labels


class Data(Dataset):
    """Dataset base class.

    Provides an interface and some default implementations for fetching, organizing, and representing external
    datasets. Designed to be incrementally extended.

    Parameters
    ----------
    identifier: str, optional
        Name for this dataset.

    root: str, optional
        Local root directory for this dataset.

    remote_urls: str or list of str, optional
        File-wise remote URL(s) from which to fetch this dataset, if applicable.

    samples: Tensor, optional
        A buffer holding data samples in tensor form. Useful for initializing Data objects on the fly, e.g.
        during data synthesis. Users may leverage :meth:`access_data` to manage disk I/O, e.g.
        using :class:`torch.storage.Storage`, memory mapped arrays/tensors, or HDF5 lazy loading.

    download: bool, optional
        Whether to download the set regardless of whether it already exists in the local directory.
        Defaults to `False`, in which case the download only commences if the `root` doesn't exist
        or is empty.

    See Also
    --------
    `PyTorch Storage Class <https://discuss.pytorch.org/t/memory-mapped-tensor/8954>`_

    """

    def __init__(
        self,
        identifier: str = None,
        root: str = None,
        remote_urls: str = None,
        samples: Tensor = None,
        download: bool = False,
        **kwargs,
    ):
        # dataset identifier, sometimes used with default methods.
        self.identifier = identifier

        # local and remote directories.
        self.root = root
        self.remote_urls = remote_urls

        # optional, data tensor provided by user at instantiation.
        self.samples = samples

        # container for axis descriptors (label vectors describing specific data axes).
        self.descriptors = {}

        # fetch dataset from remote URL if `root` doesn't exist or is empty.
        download_required = bool(self.root) and (
            not os.path.exists(self.root) or len(os.listdir(os.path.dirname(self.root))) == 0
        )

        if download or download_required:
            self._download()

            # converts the foreign dataset to a single file in a user-determined format.
            # passes silently if not implemented by the user.
            self._standardize()

        # add descriptor labels from kwargs if provided (path to a CSV file or AxisDescriptor object(s)).
        self.add_descriptors(**kwargs)

    def __call__(self):
        """Calling a dataset will invoke :meth:`load` and return a self reference."""
        self.load()
        return self

    def __getitem__(self, index: int | tuple[int]):
        """Utilizes the user-specified :meth:`access_data` to slice into the data or access specific file(s),
        returning the value(s) at `index`."""
        return self.access_data(index)

    def __len__(self):
        """The default implementation addresses trivial cases where the set is loaded to memory (in `self.samples`).
        Users should override this method when dealing with large datasets, e.g. ones that live in
        gigantic HDF5 files that are dynamically loaded."""
        return len(self.samples)

    def _download(self):
        """Downloads one or more files to the `root` directory from `remote_urls`.

        Raises
        ------
        TypeError
            If a URL is not provided and the data does not exist locally at `root`.

        ValueError
            If the URL provided is invalid.

        """
        # create the destination directory if it doesn't exist.
        ensure_dir(self.root)

        # if given a single URL, wrap it in a list.
        if isinstance(self.remote_urls, str):
            self.remote_urls = [self.remote_urls]

        try:
            # download file(s).
            for url in self.remote_urls:
                file = url.split("/")[-1]
                urllib.request.urlretrieve(url, os.path.join(self.root, file))

        except TypeError as e:
            raise TypeError(e)

        except ValueError as e:
            raise ValueError(e)

    def _standardize(self):
        """Standardizes an external data directory tree, e.g. a remote repository, possibly converting it to a
        single condensed file at `root` for subsequent operations. Should only be performed once after fetching
        the data from its remote source, if applicable.

        This method is meant to be implemented by child classes that deal with arbitrarily formatted external sets.
        It may also serve as a bridge between Sapicore and specialized libraries, e.g. MNE for EEG/MEG data.

        Warning
        -------
        The resulting file may follow any convention the user would like to adopt. Sapicore makes no assumptions;
        it is the user's responsibility to implement this method and :meth:`access_data` below in a compatible way.

        """
        pass

    def access_data(self, index: int | list[int] | tuple[int] | Tensor, axis: int = None):
        """Specifies how to access data by mapping indices to actual samples (e.g., from file(s) in `root`).

        The default implementation slices into `self.samples` to accommodate the trivial cases where the user has
        directly initialized this :class:`Data` object with a `samples` tensor or loaded its values by reading
        a file that fits in memory (the latter case would be handled by :meth:`load`).

        More sophisticated use cases may require lazy loading or navigating HDF5 files. That kind of logic should
        be implemented here by derivative classes.

        Parameters
        ----------
        index: int or tuple of int or list of int or Tensor
            Index(es) to slice into.

        axis: int, optional
            Optionally, a specific axis along which to apply index selection.

        Note
        ----
        Where audio/image files are concerned (each being a labeled "sample"), use this method to read
        and potentially transform them, returning the finished product.

        """
        if axis:
            return self.samples.index_select(axis, torch.as_tensor(index))
        else:
            return self.samples[index]

    def add_descriptors(self, file: str = None, include: list[str] = None, **kwargs: AxisDescriptor):
        """Extracts labels from file(s) in the data `root` directory, if given, and adds named
        :class:`AxisDescriptor` objects to this :class:`Data` set.

        This default implementation is intended to capture the general case where descriptor variables are
        given in the header of a tabular CSV and values in each row describe the samples (rows) of the
        associated :class:`Data` object. User may specify which column names to include.

        Parameters
        ----------
        file: str
            Path to the file containing label information.

        include: list of str
            List of keys to generate label vectors from (e.g., column names in a CSV).

        kwargs: AxisDescriptor
            Directly add one or more named :class:`AxisDescriptor` to this :class:`Data` object.

        """
        # add labels from file if given, wrapping them in a descriptor object.
        if file and os.path.exists(file):
            with open(file) as f:
                table = pd.read_csv(f)
                for c in table.columns:
                    if not bool(include) or c in include:
                        self.descriptors[c] = AxisDescriptor(name=c, labels=table[c].tolist())

        # add descriptors from extra AxisDescriptor arguments.
        for key, value in kwargs.items():
            self.descriptors[key] = value

    def aggregate_descriptors(self, axis: int = 0) -> pd.DataFrame:
        """Condenses the referenced :class:`~Data` set's :class:`~AxisDescriptor` objects into a
        pandas dataframe `table`. Performed for row (sample) descriptors by default, but can also be
        used to aggregate :class:`AxisDescriptor` objects describing any other `axis` of this dataset.

        """
        df = pd.DataFrame()
        for key, values in self.descriptors.items():
            if values.axis == axis:
                df[key] = values.labels

        return df

    def load(self, indices: tuple[int] | list[tuple[int]] = None, kind: str = None):
        """Populates the `samples` tensor buffer and/or `descriptors` attribute table by loading one or more files
        into memory, potentially selecting only `indices`. Since different datasets have different formats,
        users will have to override this method.

        Parameters
        ----------
        indices: tuple of int or list of tuple of int
            Specific indices to include, one for each file.

        kind: str
            Kind of content to load: "data" or "descriptors"/"labels" (defaults to both).

        Note
        ----
        When it comes to data I/O, each dataset will have be treated differently, depending on size and
        other considerations. It is up to the user to decide what is best for their particular dataset.

        Warning
        -------
        Populating the `samples` buffer with the entire dataset should only be done when it can fit in memory.
        For large sets, the buffer should not be used directly or naively; :meth:`access_data` should be overriden
        to implement some form of lazy loading.

        """
        pass

    def save(self, destination: str, kind: str = None):
        """Dump the contents of the `samples` buffer and/or `descriptors` attribute table to disk at `destination`.
        Since different datasets have different formats, users may wish to override this method.

        While the default implementation saves the tensor as a .pt file, users are encouraged to override
        this method for every dataset and pick/design a format appropriate for their data.

        Parameters
        ----------
        destination: str
            Path to the desired output file.

        kind: str
            Kind of content to save: "data" or "descriptors"/"labels" (saves everything by default).

        """
        if kind is None or kind == "descriptors" or kind == "labels":
            if destination.endswith(".csv"):
                table = self.aggregate_descriptors()
                pd.to_csv(table, ensure_dir(destination))

        if kind is None or kind == "data":
            if destination.endswith(".pt"):
                torch.save(self.samples, ensure_dir(destination))

            elif destination.endswith(".csv"):
                pd.to_csv(self.samples, ensure_dir(destination))

    def scan_root(self, pattern: str = "*", recursive: bool = False) -> list[str]:
        """Scans the `root` directory and returns a list of files found that match the glob `pattern`
        (similar to a regex).

        Parameters
        ----------
        pattern: str, optional
            Pattern to look for in file names that should be included (glob formatted).

        recursive: bool, optional
            Whether glob should search recursively. Defaults to `False`.

        """
        files = []
        if os.path.exists(self.root):
            for f in glob(os.path.join(self.root, pattern), recursive=recursive):
                files.append(f)

        return natsorted(files)

    def select(self, conditions: str | list[str], axis: int = 0) -> list[int]:
        """Selects sample indices by descriptor values based on a pandas logical statement applied to one or more
        :class:`AxisDescriptor` objects, aggregated into a dataframe using :meth:`aggregate_descriptors`.

        Importantly, these filtering operations can be performed prior to loading any data to memory, as they only
        depend on :class:`AxisDescriptor` objects (labels) attached to this dataset.

        Note
        ----
        Applying selection criteria with pd.eval syntax is simple:

        * "age > 21" would execute `pd.eval("self.table.age > 21", target=df)`.
        * "lobe.isin['F', 'P']" would execute `pd.eval("self.table.lobe.isin(['F', 'P']", target=df)`.

        Where `df` is the table attribute of the :class:`~AxisDescriptor` the condition is applied to.

        """
        # wrap a single string condition in a list if necessary.
        if isinstance(conditions, str):
            conditions = [conditions]

        # aggregate axis descriptors into a pandas dataframe (table).
        df = self.aggregate_descriptors(axis=axis)

        # evaluate the expressions and filter the dataframe.
        for expression in conditions:
            df = df[pd.eval("df." + expression, target=df).to_list()]

        # return indices where expressions evaluated to true.
        return df.index.to_list()

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
            Axis along which selection is performed. Defaults to zero (that is, rows/samples).

        kwargs:
            retain: int or float
                The number or proportion of samples to retain.

            shuffle: bool
                If using a :class:`sklearn.model_selection.BaseCrossValidator` to sample,
                whether to toggle the `shuffle` parameter on.

            label_keys: str or list of str
                Label key(s) by which to stratify sampling, if applicable.

            group_key: str
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
            iterator = CV(data=self, cross_validator=validator, **kwargs)

            # use the first fold.
            _, subset = next(iter(iterator))

        elif isinstance(method, BalancedSampler):
            # special case of the balanced sampler, which uses the aggregated descriptor dataframe.
            subset = method(frame=self.aggregate_descriptors(), **kwargs)

        else:
            # in all other cases, directly apply the method provided.
            subset = method(**kwargs)

        # trim samples and labels, returning a new partial dataset without mutating the original.
        return self.trim(index=subset, axis=axis)

    def trim(self, index: int | list[int] | tuple[int] | Tensor, axis: int = 0):
        """Trims this set based on `index` and `axis`, returning a subset thereof in terms of both data and
        labels/descriptors. Does not mutate the underlying object."""

        # create a deep copy of this data object and mutate it.
        partial_data = deepcopy(self)
        partial_data.samples = self.access_data(index, axis=axis)

        for k in partial_data.descriptors.keys():
            if partial_data.descriptors[k].axis == axis:
                partial_data.descriptors[k].labels = np.array(partial_data.descriptors[k].labels)[index].tolist()

        return partial_data
