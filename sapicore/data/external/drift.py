"""UCSD Drift dataset."""
from zipfile import ZipFile

import os
import shutil

import torch
import pandas as pd

from scipy.sparse import csr_matrix as csr
from sklearn.datasets import load_svmlight_file

from sapicore.data import Data, AxisDescriptor, Metadata
from sapicore.utils.io import ensure_dir


class DriftDataset(Data):
    """Processing logic for the UCSD drift dataset.

    Implements the protected :meth:`~data.Data._standardize` to extract, transform, and save buffer and labels
    from the raw .dat files downloaded from the UCI repository.

    See Also
    --------
    `Drift dataset specification <https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset>`_

    """

    def __init__(self, root: str, **kwargs):
        super().__init__(
            root=root,
            download=True,
            remote_urls="https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
            **kwargs
        )

    def _standardize(self):
        # the file retrieved from the UCSD repository needs to be unzipped.
        with ZipFile(os.path.join(self.root, "gas+sensor+array+drift+dataset.zip"), "r") as zf:
            zf.extractall(path=os.path.join(self.root))

        # remove zip archive and move extracted files to the dataset root directory "drift".
        os.remove(os.path.join(self.root, "gas+sensor+array+drift+dataset.zip"))
        for file in os.listdir(os.path.join(self.root, "Dataset")):
            shutil.move(os.path.join(self.root, "Dataset", file), os.path.join(self.root, file))
        shutil.rmtree(os.path.join(self.root, "Dataset"))

        # scan the root directory for .dat files, which happen to be in libsvm format.
        raw_files = self.scan(pattern="*.dat")

        # placeholder for accumulating label vectors.
        meta = {"chemical": [], "batch": []}

        for i, file in enumerate(raw_files):
            # load and convert from sparse matrix to standard numpy array.
            data, labels = load_svmlight_file(file)
            data = csr(data, dtype=data.dtype).todense()

            # since the dataset is small, vertically stack batch-specific data onto the `buffer`.
            self.buffer = (
                torch.tensor(data, dtype=torch.float64)
                if i == 0
                else torch.vstack((self.buffer, torch.tensor(data, dtype=torch.float64)))
            )

            meta["chemical"].extend(labels)
            meta["batch"].extend([i + 1 for _ in range(len(labels))])

        # add metadata to dataset.
        self.metadata = Metadata(
            AxisDescriptor(name="chemical", labels=meta["chemical"], axis=0),
            AxisDescriptor(name="batch", labels=meta["batch"], axis=0),
        )

        # dump the data and label tensors to .pt for loading convenience.
        self.save()

    def load(self, indices: slice = None):
        """Load drift data and labels to memory from the .pt files saved by :meth:`~data.Data._standardize`.
        Overwrites `buffer` and `descriptors`. If specified, selects only the rows at `indices`.

        Parameters
        ----------
        indices: slice, optional
            Specific indices to load.

        """
        # the labels file has been created by _standardize upon first obtaining the data.
        labels = torch.load(os.path.join(self.root, "labels.pt"), weights_only=False)
        if indices is not None:
            labels = labels[indices]

        # for the drift set specifically, we know that all labels describe the 0th axis (rows).
        self.metadata = Metadata(*[AxisDescriptor(name=col, labels=labels[col].to_list(), axis=0) for col in labels])

        # load the data into the buffer (since the drift set is small, no need for lazy loading).
        self.buffer = torch.load(os.path.join(self.root, "data.pt"), weights_only=False)

        if indices is not None:
            self.buffer = self.buffer[indices]

        return self

    def save(self):
        """Dump the contents of the `buffer` and `metadata` table to disk at `destination`.
        This implementation saves the tensors as .pt files.

        """
        torch.save(self.buffer, os.path.join(ensure_dir(self.root), "data.pt"))
        torch.save(self.metadata.to_dataframe(), os.path.join(ensure_dir(self.root), "labels.pt"))

    def select(self, conditions: str | list[str], axis: int = 0) -> slice:
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

        Warning
        -------
        This method is general and may eventually be moved to the base :class:`~data.Data` class.

        """
        # wrap a single string condition in a list if necessary.
        if isinstance(conditions, str):
            conditions = [conditions]

        # aggregate relevant axis descriptors into a pandas dataframe (table).
        df = self.metadata.to_dataframe(axis=axis)

        # evaluate the expressions and filter the dataframe.
        for expression in conditions:
            parsed = "& ".join([("df." + expr).replace(". ", ".") for expr in expression.split("&")])
            parsed = "| ".join([("df." + expr).replace(". ", ".") for expr in parsed.split("|")])

            parsed = parsed.replace("df.df.", "df.")
            parsed = parsed.replace("df.(", "(df.")

            df = df[pd.eval(parsed, target=df).to_list()]

        # return indices where expressions evaluated to true.
        return df.index.to_list()
