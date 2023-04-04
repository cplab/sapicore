"""UCSD Drift dataset."""
import os
import shutil

from zipfile import ZipFile

import torch
import pandas as pd

from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix as csr

from sapicore.data import Data, AxisDescriptor


class DriftDataset(Data):
    """Processing logic for the UCSD drift dataset.

    Implements the protected :meth:`~data.Data._standardize` to extract, transform, and save samples and labels
    from the raw .dat files downloaded from the UCI repository.

    See Also
    --------
    `Drift dataset specification <https://archive.ics.uci.edu/ml/datasets/Gas%20Sensor%20Array%20Drift%20Dataset#>`_

    """

    def __init__(self, root: str, **kwargs):
        super().__init__(
            root=root,
            remote_urls="https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip",
            **kwargs
        )

    def _standardize(self):
        # the file retrieved from the UCSD repository needs to be unzipped.
        with ZipFile(os.path.join(self.root, "Dataset.zip"), "r") as zf:
            zf.extractall(path=os.path.join(self.root))

        # remove zip archive and move extracted files to the dataset root directory "drift".
        os.remove(os.path.join(self.root, "Dataset.zip"))
        for file in os.listdir(os.path.join(self.root, "Dataset")):
            shutil.move(os.path.join(self.root, "Dataset", file), os.path.join(self.root, file))
        shutil.rmtree(os.path.join(self.root, "Dataset"))

        # scan the root directory for .dat files.
        raw_files = self.scan_root(pattern="*.dat")

        # the .dat files are in libsvm format and shall be merged into a single dataframe and saved to CSV,
        # with an extra row specifying the batch they came from.
        unified_data = []

        # initialize an empty axis descriptor for the single label vector given in the drift dataset.
        self.add_descriptors(chemical=AxisDescriptor(name="chemical", labels=[], axis=0))
        self.add_descriptors(batch=AxisDescriptor(name="batch", labels=[], axis=0))

        for i, file in enumerate(raw_files):
            # load and convert from sparse matrix to standard numpy array.
            data, labels = load_svmlight_file(file)

            data = csr(data, dtype=data.dtype).todense()
            unified_data.append(pd.DataFrame(data))

            # since the dataset is small, we shall add each batch to the `samples` buffer.
            self.samples = (
                torch.tensor(data, dtype=torch.float64)
                if i == 0
                else torch.vstack((self.samples, torch.tensor(data, dtype=torch.float64)))
            )

            self.descriptors["chemical"].labels.extend(labels)
            self.descriptors["batch"].labels.extend([i + 1 for _ in range(len(labels))])

        # concatenate the data frames and add a batch identifier column.
        unified_data = pd.concat(unified_data, keys=[i + 1 for i in range(len(raw_files))]).reset_index()
        unified_data = unified_data.rename(
            columns={unified_data.level_0.name: "batch", unified_data.level_1.name: "sample"}
        )

        # dump the resulting frame and labels to CSV for human readability.
        unified_data["chemical"] = self.descriptors["chemical"].labels
        unified_data["batch"] = self.descriptors["batch"].labels

        unified_data.to_csv(os.path.join(self.root, "data.csv"), index=False)

        # dump the data and label tensors to .pt for loading convenience.
        torch.save(self.samples, os.path.join(self.root, "data.pt"))
        torch.save(self.aggregate_descriptors(), os.path.join(self.root, "labels.pt"))

    def load(self, indices: tuple[int] | list[tuple[int]] = None, kind: str = None):
        """Load drift data and labels to memory from the .pt files saved by :meth:`~data.Data._standardize`.
        Overwrites `samples` and `descriptors`.

        """
        if kind is None or kind == "descriptors" or kind == "labels":
            self.descriptors = {}

            labels = torch.load(os.path.join(self.root, "labels.pt"))
            if indices is not None:
                labels = labels[indices]

            # for the drift set specifically, we know that all labels describe the 0th axis (rows, samples).
            self.add_descriptors(
                **{col: AxisDescriptor(name=col, labels=labels[col].to_list(), axis=0) for col in labels}
            )

        if kind is None or kind == "data":
            self.samples = torch.load(os.path.join(self.root, "data.pt"))
            if indices is not None:
                self.samples = self.samples[indices]
