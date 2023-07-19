import pytest
import os

from sapicore.data.external.drift import DriftDataset
from sapicore.tests import ROOT

TEST_ROOT = os.path.join(ROOT, "tests", "data", "test_data")  # test root directory.


class TestDrift:
    @pytest.mark.unit
    def test_drift_processing(self):
        # calling a dataset invokes its load() method and returns a self reference.
        try:
            DriftDataset(root=os.path.join(TEST_ROOT, "drift")).load()

        except ConnectionError:
            # package tests should NOT fail when a particular hardcoded archive server is down.
            pytest.mark.skip("Could not download drift dataset")


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
