import pytest
import os

from sapicore.data.external.drift import DriftDataset
from sapicore.tests import ROOT

TEST_ROOT = os.path.join(ROOT, "tests", "data", "test_data")  # test root directory.


class TestDrift:
    @pytest.mark.unit
    def test_drift_processing(self):
        # calling a dataset invokes its load() method and returns a self reference.
        drift = DriftDataset(
            identifier="drift",
            root=os.path.join(TEST_ROOT, "drift"),
            remote_urls="https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip",
        )()

        # prove that data was correctly processed.
        print(f"Drift samples shape: {drift.samples.shape}")
        print(f"Drift labels shape: {drift.aggregate_descriptors().shape}")


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
