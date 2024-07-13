""" Ensemble unit tests. """
import os
import pytest

import torch

from sapicore.engine.ensemble.analog import AnalogEnsemble, OscillatorEnsemble
from sapicore.engine.ensemble.spiking import LIFEnsemble, IZEnsemble

from sapicore.tests import ROOT, TEST_DEVICE


STEPS = 100  # default number of simulation forward steps to execute during testing.
TEST_ROOT = os.path.join(ROOT, "tests", "ensemble", "test_ensemble")  # test root directory.


class TestEnsemble:
    @pytest.mark.parametrize(
        "arg_",
        [
            AnalogEnsemble(device=TEST_DEVICE),
            AnalogEnsemble(num_units=1000, device=TEST_DEVICE),
        ],
        ids=["DEFAULT", "MULTIUNIT"],
    )
    @pytest.mark.unit
    def test_analog(self, arg_: AnalogEnsemble | OscillatorEnsemble):
        """Task initialization test cases."""
        # verify that all attributes are initialized.
        assert all(hasattr(arg_, attr) for attr in ["identifier", "input", "voltage", "num_units"])

        # verify that loggable attributes are torch tensors on the correct device.
        assert all(type(getattr(arg_, attr)) is torch.Tensor for attr in arg_._loggable_props_)
        assert all(getattr(arg_, attr).device == torch.device(TEST_DEVICE) for attr in arg_._loggable_props_)

        # verify behavior of trivial forward method.
        init_voltage = arg_.voltage

        # adding a compatible shape data should increase `voltage` buffer values.
        add_same_shape = arg_.forward(data=torch.ones_like(init_voltage, device=TEST_DEVICE))
        assert torch.all(add_same_shape["voltage"].eq(init_voltage + 1))
        assert torch.all(add_same_shape["voltage"].eq(arg_.voltage))

    @pytest.mark.parametrize(
        "arg_",
        [
            LIFEnsemble(device=TEST_DEVICE),
            IZEnsemble(num_units=1000, device=TEST_DEVICE),
        ],
        ids=["LIF", "IZ"],
    )
    @pytest.mark.unit
    def test_generic(self, arg_: LIFEnsemble | IZEnsemble):
        # TODO complete unit tests and add cases as necessary.
        pass


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
