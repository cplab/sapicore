"""UCSD gas sensor drift dataset
================================

"""
import pandas as pd
import numpy as np
import os
import torch
from torch.utils import data

__all__ = ('UCSDGasDrift', 'Datasplitter')


class UCSDGasDrift(data.Dataset):
    """UCSD gas sensor  drift dataset.
    Dataset has been split into batches. Each batch has 3 files:
    "conc.csv": Gas concentration
    "data.csv": Raw peak sensor responses
    "y.csv": Six gas types
    TODO: Automatic data download from web and preprocess.
    """

    def __init__(self, list_ids: list, labels: list,
                 root_dir='data/drift/', batch_idx=1):
        """UCSD gas sensor dataset.

        :param filetype: path to the csv file.
        :param root_dir: Directory with all the csv files
        """
        self.labels = labels
        self.list_ids = list_ids
        self.scaled_data = pd.read_csv(
            root_dir + os.sep + 'uscaled_array.csv',
            header=None).values.flatten()
        self.data = pd.read_csv(
            root_dir + os.sep + 'batch' + str(batch_idx) + os.sep +
            'data.csv', header=None)

    def __len__(self) -> int:
        return len(self.list_ids)

    def __getitem__(self, index) -> list:

        id = self.list_ids[index]
        # Load data and get label
        gas_sensor_data = self.data.iloc[id, :].values
        gas_sensor_data /= self.scaled_data  # scale sensor data.
        # Might use torch transform later
        y = self.labels[id]

        return [torch.tensor(gas_sensor_data, dtype=torch.float), y]


class Datasplitter:
    """Splits the drift dataset into multiple shots.
    """

    def __init__(self, root_dir: str, validation=True, **kwargs):
        self.num_shots = kwargs["num_shots"]
        self.isvalidationset = validation
        self.y_labels = pd.read_csv(root_dir + '/' + 'y.csv',
                                    header=None).values.flatten()
        self.class_indices = np.unique(self.y_labels)
        self.num_groups = len(self.class_indices)

    def __call__(self) -> tuple:
        partition = {}
        labels = {}

        for grp_idx in range(1, self.num_groups + 1):
            partition[grp_idx] = {}
            allidsforgroup, = np.where(self.y_labels == grp_idx)
            if grp_idx == 1:
                for id in range(len(self.y_labels)):
                    labels[id] = self.y_labels[id]
            train_ids = np.random.choice(np.where(self.y_labels == grp_idx)[0],
                                         self.num_shots, replace=False).tolist()
            partition[grp_idx]['train'] = train_ids

            allidsforgroup = np.setdiff1d(allidsforgroup, train_ids)
            if self.isvalidationset:
                val_ids = np.random.choice(allidsforgroup,
                                           int(0.1 * len(allidsforgroup)),
                                           replace=False)
                if grp_idx > 1:
                    partition[grp_idx][
                        'validation'] = partition[grp_idx - 1]['validation'] + \
                                        val_ids.tolist()
                else:
                    partition[grp_idx]['validation'] = val_ids.tolist()
                allidsforgroup = np.setdiff1d(allidsforgroup, val_ids)
            test_ids = allidsforgroup
            if grp_idx > 1:
                partition[grp_idx]['test'] = partition[grp_idx - 1]['test'] + \
                                             test_ids.tolist()
            else:
                partition[grp_idx]['test'] = test_ids.tolist()
        return partition, labels, len(self.class_indices)
