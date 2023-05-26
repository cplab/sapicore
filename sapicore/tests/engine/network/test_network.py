""" Network unit and functional tests. """
import os
import pytest

from sapicore.pipeline.simple import SimpleSimulator
from sapicore.tests import ROOT

TEST_ROOT = os.path.join(ROOT, "tests", "engine", "network", "test_network")  # test root directory.


class TestNetwork:
    @pytest.mark.parametrize("cfg_", [os.path.join(TEST_ROOT, "example.yaml")], ids=["SYNTH"])
    @pytest.mark.functional
    @pytest.mark.slow
    def test_network(self, cfg_: dict):
        """All YAML-parameterized network tests go here."""
        SimpleSimulator(configuration=cfg_).run()


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
