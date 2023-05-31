""" Synapse unit and integration tests. """
import os
import pytest

from sapicore.engine.synapse import Synapse
from sapicore.engine.synapse.STDP import STDPSynapse

from sapicore.tests import ROOT

STEPS = 100  # default number of simulation forward steps to execute during testing.
TEST_ROOT = os.path.join(ROOT, "tests", "synapse", "test_synapse")  # test root directory.


class TestSynapse:
    @pytest.mark.parametrize("arg_", [Synapse(), STDPSynapse()], ids=["BASE", "STDP"])
    @pytest.mark.unit
    def test_init(self, arg_: Synapse):
        """Initialization test cases."""
        # TODO write simple unit and integration tests.
        pass


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
