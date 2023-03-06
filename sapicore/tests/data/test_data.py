""" Data test suite. """
import pytest
import os

import torch
from sklearn.model_selection import StratifiedGroupKFold

from sapicore.data import Data, AxisDescriptor
from sapicore.data.sampling import CV

from sapicore.utils.sweep import Sweep
from sapicore.tests import ROOT

from sapicore.engine.network import Network
from sapicore.model import Model

TEST_ROOT = os.path.join(ROOT, "tests", "data", "test_data")  # test root directory.


class TestData:
    @pytest.mark.unit
    def test_data_pipeline(self):
        # initialize a dummy dataset with balanced labels.
        data = Data(samples=torch.rand(16, 4))
        params = Sweep({"grid": {"study": [1, 2], "animal": ["A", "B", "C", "D"]}}, 8)(dataframe=True)

        study = AxisDescriptor(name="study", labels=params["study"].tolist(), axis=0)
        animal = AxisDescriptor(name="animal", labels=params["animal"].tolist(), axis=0)
        sensor = AxisDescriptor(name="sensor", labels=["Occipital", "Parietal", "Temporal", "Frontal"], axis=1)

        # register the above axis descriptors (label vectors) with our Data object.
        data.add_descriptors(study=study, animal=animal, sensor=sensor)
        table = data.aggregate_descriptors()

        # demonstrate logical selection of sample indices based on label values.
        data.select_cond(["study.isin([1])", "animal=='A'"])

        # set up a cross validation object, leveraging scikit-learn.
        cv = CV(data=data, cross_validator=StratifiedGroupKFold(2, shuffle=True), label_key="animal", group_key="study")

        # container for models trained in each CV fold.
        models = []

        for i, (train, test) in enumerate(iter(cv)):
            # initialize a new model for this cross validation fold and append it to the list.
            # a dummy network is used, but a real one could be initialized by providing a `configuration` dict.
            models.append(Model(Network()))

            # data from a particular fold would be accessed with `data[tr]` or `data[te]`.
            models[i].fit(data[train], repetitions=torch.randint(low=2, high=7, size=(data[train].shape[0],)))

            # verify correctness of cross validation folds.
            print(f"\n\nFold {i+1}\n------")
            print(f"Train: {table.iloc[train]}")
            print(f"*********\nTest: {table.iloc[test]}")

    @pytest.mark.parametrize(
        "url_",
        [
            "ftp://www.sccn.ucsd.edu/pub/eeglab_data.set",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip",
        ],
        ids=["UCSD.EEG", "UCSD.DRIFT"],
    )
    @pytest.mark.unit
    def test_fetch(self, url_):
        Data(remote_url=url_, root=os.path.join(TEST_ROOT, os.path.basename(url_)))


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])