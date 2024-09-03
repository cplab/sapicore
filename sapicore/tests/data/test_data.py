import pytest
import os

import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from sapicore.data import Data, AxisDescriptor
from sapicore.data.sampling import BalancedSampler, CV

from sapicore.utils.sweep import Sweep
from sapicore.tests import ROOT

from sapicore.engine.network import Network
from sapicore.model import Model

TEST_ROOT = os.path.join(ROOT, "tests", "data", "test_data")  # test root directory.


class TestData:
    @pytest.mark.unit
    def test_data_pipeline(self):
        # initialize a dummy dataset with balanced labels.
        data = Data(buffer=torch.rand(8, 4))
        params = Sweep({"grid": {"study": [1, 2], "animal": ["A", "B", "C", "D"]}}, 8)(dataframe=True)

        study = AxisDescriptor(name="study", labels=params["study"].tolist(), axis=0)
        animal = AxisDescriptor(name="animal", labels=params["animal"].tolist(), axis=0)
        sensor = AxisDescriptor(name="sensor", labels=["Occipital", "Parietal", "Temporal", "Frontal"], axis=1)

        # register the above axis descriptors (label vectors) with our Data object.
        data.metadata.add_descriptors(study, animal, sensor)

        # print the row descriptors in tabular form.
        data.metadata.to_dataframe(axis=0)

        # different ways of accessing label vectors and data with slicing.
        _ = data.metadata["sensor"][:]
        _ = data[:]

        # demonstrate stratified sampling w.r.t. one descriptor using a base cross validator.
        data.sample(method=StratifiedKFold, shuffle=True, retain=4, label_keys="animal")

        # demonstrate balanced sampling w.r.t. two descriptors using a BalancedSampler (1 for each crossing).
        data.sample(method=BalancedSampler(replace=False, stratified=False), group_keys=["animal", "study"], n=1)

        # set up a cross validation object, leveraging scikit-learn.
        cv = CV(
            data=data, cross_validator=StratifiedGroupKFold(2, shuffle=True), label_keys="animal", group_key="study"
        )

        # container for models trained in each CV fold.
        models = []

        for i, (train, test) in enumerate(cv):
            # initialize a new model for this cross validation fold and append it to the list.
            # a dummy network is used, but a real one could be initialized by providing a `configuration` dict.
            models.append(Model(Network()))

            # repeats each sample for a random number of steps (simulating variable exposure durations).
            models[i].fit(data[train], duration=torch.randint(low=2, high=7, size=(data[train].shape[0],)))

    @pytest.mark.parametrize(
        "url_",
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip",
        ],
        ids=["DRIFT"],
    )
    @pytest.mark.unit
    def test_fetch(self, url_):
        Data(remote_urls=url_, root=os.path.join(TEST_ROOT, os.path.basename(url_).split(".")[0]))


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
