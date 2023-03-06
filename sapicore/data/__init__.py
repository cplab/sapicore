""" Data operations. """
import os
import urllib.request

from glob import glob

import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset

from sapicore.utils.io import ensure_dir

__all__ = ("AxisDescriptor", "Data")


class AxisDescriptor:
    """Describes one axis of an N-dimensional :class:`Data` array by a vector of `labels`.

    Parameters
    ----------
    name: str, optional
        Name of the descriptor (i.e., the variable whose value is given by the labels).

    labels: list, optional
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

    def __init__(self, name: str = None, labels: list = None, axis: int = 1):
        self.name = name
        self.axis = axis
        self.labels = labels

    def __str__(self):
        return str(self.labels)


class Data(Dataset):
    """Dataset base class.

    Provides an interface and some default implementations for fetching, organizing, and representing external
    datasets. Designed to be incrementally extended.

    Parameters
    ----------
    identifier: str, optional
        Name for this dataset.

    samples: Tensor, optional
        A buffer holding data samples in tensor form. Useful for initializing Data objects on the fly, e.g.
        during data synthesis. Users may leverage :meth:`access_data` to manage disk I/O, e.g.
        using :class:`torch.Storage`, memory mapped arrays/tensors, or HDF5 lazy loading.

    root: str, optional
        Local root directory for this dataset.

    remote_url: str, optional
        Remote URL from which to fetch this dataset, if applicable.

    force_download: bool, optional
        Whether to force download even if the set is in the local directory. Defaults to `False`.

    See Also
    --------
    `PyTorch Storage Class<https://discuss.pytorch.org/t/memory-mapped-tensor/8954>`_

    """

    def __init__(
        self,
        identifier: str = None,
        root: str = None,
        remote_url: str = None,
        samples: Tensor = None,
        force_download: bool = False,
        **kwargs: AxisDescriptor,
    ):
        # dataset identifier, sometimes used with default methods.
        self.identifier = identifier

        # local and remote directories.
        self.root = root
        self.remote_url = remote_url

        # optional, data tensor provided by user at instantiation.
        self.samples = samples

        # fetch dataset from remote URL if `root` doesn't exist or is empty.
        download_required = bool(self.root) and (
            not os.path.exists(self.root) or len(os.listdir(os.path.dirname(self.root))) == 0
        )

        if force_download or download_required:
            # TODO add integrity checks.
            self._download()

            # converts the foreign dataset to a single file in a user-determined format, e.g. hdf5 or npz.
            # passes silently if not implemented by the user.
            self._standardize()

        # label vectors describing particular axes.
        self.descriptors = {}
        self.add_descriptors(**kwargs)

    def __getitem__(self, index: int | tuple[int]):
        """Utilizes the user-specified :meth:`read_data` to slice into the data or access specific file(s),
        returning the value(s) at `index`."""
        return self.access_data(index)

    def __len__(self):
        """The default implementation addresses trivial cases where the set is loaded to memory (in `self.samples`).

        Users should override this method when dealing with large datasets, e.g. ones that live in
        gigantic HDF5 files that are dynamically loaded.

        """
        return len(self.samples)

    def _download(self):
        """Downloads a dataset to the `root` directory from `remote_url`."""
        # create the destination file directory if it doesn't exist.
        ensure_dir(os.path.dirname(self.root))

        try:
            # download file from remote URL.
            urllib.request.urlretrieve(self.remote_url, self.root)

        except ValueError:
            print(f"Could not fetch data, invalid URL ({self.remote_url}).")

    def _standardize(self):
        """Standardizes an external data directory tree, e.g. a remote repository, possibly converting it to a
        single file at `root` for subsequent operations. Should only be performed once after fetching the data
        from its remote source, if applicable.

        This method should be implemented by child classes that deal with arbitrarily formatted external sets.
        It may also serve as a bridge between Sapicore and specialized libraries, e.g. MNE for EEG/MEG data.

        Warning
        -------
        The resulting file may follow any convention the user would like to adopt. Sapicore makes no assumptions;
        it is the user's responsibility to implement this method and :meth:`access_data` below in a compatible way.

        """
        pass

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
        """Condenses the referenced :class:`~data.Data` set's :class:`~AxisDescriptor` objects into a
        pandas dataframe `table`. Performed for row (sample) descriptors by default, but can also be
        used to aggregate :class:`AxisDescriptor` objects describing any other `axis` of this dataset.

        """
        df = pd.DataFrame()
        for key, values in self.descriptors.items():
            if values.axis == axis:
                df[key] = values.labels

        return df

    def load_data(self, files: str | list[str] = None, indices: tuple[int] | list[tuple[int]] = None):
        """Populates the `samples` tensor buffer by loading data from one or more `files` into memory,
        potentially selecting only `indices`.

        Parameters
        ----------
        files: str or list of str
            One or more paths to data files.

        indices: tuple of int or list of tuple of int
            Specific indices to include, one for each file.

        Warning
        -------
        Populating the `samples` buffer with the entire dataset should only be done when it can fit in memory.
        For large sets, :meth:`access_data` should be overriden to implement some form of lazy loading.

        """
        pass

    def access_data(self, index: int | tuple[int]):
        """Specifies how to access data by mapping indices to actual samples (e.g., from file(s) in `root`).

        The default implementation slices into `self.samples` to accommodate the trivial cases where the user has
        directly initialized this :class:`Data` object with a `samples` tensor or loaded its values by reading
        a file that fits in memory (the latter case would be handled by :meth:`load_data`).

        More sophisticated use cases may require lazy loading or navigating HDF5 files. That kind of logic should
        be implemented here by derivative classes.

        Parameters
        ----------
        index: int or tuple of int
            Index(ices) to slice into.

        """
        return self.samples[index]

    def select_cond(self, conditions: str | list[str], axis: int = 0) -> list[int]:
        """Selects sample indices by descriptor values based on a pandas logical statement applied to one or more
        :class:`AxisDescriptor` objects, aggregated into a dataframe using :meth:`aggregate_descriptors`.

        Importantly, these filtering operations can be performed before loading any data to memory, as they only
        depend on :class:`AxisDescriptor` objects attached to this dataset.

        Note
        ----
        Applying selection criteria with pd.eval syntax is simple:

        * "age > 21" would execute `pd.eval("self.table.age > 21", target=df)`.
        * "lobe.isin['F', 'P']" would execute `pd.eval("self.table.lobe.isin(['F', 'P']", target=df)`.

        Where `df` is the table attribute of the :class:`~data.descriptor.Descriptor` the condition is applied to.

        The selection operation comes down to: array[:, table[table[<field>].isin([<values>])].index.to_list()]
        So, once filtered specific `indices` for axis=0, Data.load() just loads file[indices, :, :].

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

        return files
