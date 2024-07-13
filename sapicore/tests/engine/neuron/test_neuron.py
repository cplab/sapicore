""" Neuron unit tests. """
import os
import pytest

import torch

from sapicore.engine.neuron import Neuron
from sapicore.engine.neuron.analog import AnalogNeuron
from sapicore.engine.neuron.analog.oscillator import OscillatorNeuron
from sapicore.engine.neuron.spiking import SpikingNeuron
from sapicore.engine.neuron.spiking.LIF import LIFNeuron
from sapicore.engine.neuron.spiking.IZ import IZNeuron

from sapicore.tests import ROOT, TEST_DEVICE

STEPS = 100  # default number of simulation forward steps to execute during testing.
TEST_ROOT = os.path.join(ROOT, "tests", "neuron", "test_neuron")  # test root directory.


class TestNeuron:
    @pytest.mark.parametrize(
        "arg_",
        [
            Neuron(identifier="Test", device=TEST_DEVICE),
            SpikingNeuron(device=TEST_DEVICE),
        ],
        ids=["NEURON", "SPIKING"],
    )
    @pytest.mark.unit
    def test_base(self, arg_: Neuron):
        """Basic initialization test cases."""
        # verify that all attributes are initialized.
        assert all(
            hasattr(arg_, attr)
            for attr in (["identifier", "input", "voltage"] + ["spiked"] if isinstance(arg_, SpikingNeuron) else [])
        )

        # verify that loggable attributes are torch tensors on the correct device.
        assert all(type(getattr(arg_, attr)) is torch.Tensor for attr in arg_._loggable_props_)
        assert all(getattr(arg_, attr).device == torch.device(TEST_DEVICE) for attr in arg_._loggable_props_)

        # for coverage completeness.
        with pytest.raises(NotImplementedError):
            arg_.forward(data=torch.zeros(1))

    @pytest.mark.parametrize("arg_", [AnalogNeuron(device=TEST_DEVICE)], ids=["ANALOG"])
    @pytest.mark.unit
    def test_analog(self, arg_: AnalogNeuron):
        """Task initialization test cases."""
        # verify that all attributes are initialized.
        assert all(hasattr(arg_, attr) for attr in ["identifier", "input", "voltage"])

        # verify that loggable attributes are torch tensors on the correct device.
        assert all(type(getattr(arg_, attr)) is torch.Tensor for attr in arg_._loggable_props_)
        assert all(getattr(arg_, attr).device == torch.device(TEST_DEVICE) for attr in arg_._loggable_props_)

        # verify behavior of trivial forward method.
        init_voltage = arg_.voltage

        # adding a compatible shape data should update `voltage` buffer values.
        add_same_shape = arg_.forward(data=torch.ones_like(init_voltage, device=TEST_DEVICE))

        # adding a data tensor of a different shape should overwrite content of `voltage` buffer.
        add_diff_len = arg_.forward(
            data=torch.ones((init_voltage.shape[0] + 2, init_voltage.shape[0] + 2), device=TEST_DEVICE)
        )
        assert torch.all(add_diff_len["voltage"].eq(torch.ones_like(add_same_shape["voltage"])))
        assert torch.all(add_diff_len["voltage"].eq(arg_.voltage))

    @pytest.mark.parametrize(
        "arg_",
        [
            OscillatorNeuron(device=TEST_DEVICE),
            OscillatorNeuron(amplitudes=[5.0, 10.0], frequencies=[20.0, 40.0], device=TEST_DEVICE),
            OscillatorNeuron(amplitudes=[5.0, 10.0], frequencies=[20.0, 40.0], phases=[0.0, 0.0], device=TEST_DEVICE),
            OscillatorNeuron(
                device=TEST_DEVICE,
                amplitudes=[5.0, 10.0],
                frequencies=[20.0, 40.0],
                phases=[0.0, 0.0],
                amp_freq=[0.0, 20.0],
            ),
        ],
        ids=["DEFAULT", "AF", "AFP", "PAC"],
    )
    @pytest.mark.unit
    def test_oscillator(self, arg_: OscillatorNeuron):
        # verify that all attributes are initialized.
        assert all(hasattr(arg_, attr) for attr in ["identifier", "input", "voltage"])

        # SUGGEST add signal correctness verification of some sort.
        [arg_.forward(data=torch.ones_like(arg_.voltage, device=TEST_DEVICE)) for _ in range(STEPS)]

    @pytest.mark.parametrize(
        "arg_",
        [
            LIFNeuron(device=TEST_DEVICE),
            LIFNeuron(
                volt_thresh=-50.0,
                volt_rest=-60.0,
                leak_gl=5.0,
                tau_mem=5.0,
                tau_ref=2.0,
                device=TEST_DEVICE,
            ),
        ],
        ids=["DEFAULT", "CUSTOM"],
    )
    @pytest.mark.unit
    def test_lif(self, arg_: LIFNeuron):
        # verify that all attributes are initialized.
        assert all(hasattr(arg_, attr) for attr in ["identifier", "input", "voltage", "spiked"])

        # test forward pass
        [arg_.forward(data=torch.ones_like(arg_.voltage, device=TEST_DEVICE) * 100.0) for _ in range(STEPS)]

    @pytest.mark.parametrize(
        "arg_",
        [IZNeuron(device=TEST_DEVICE), IZNeuron(a=0.2, b=0.25, c=-60.0, d=4.0, device=TEST_DEVICE)],
        ids=["DEFAULT", "CUSTOM"],
    )
    @pytest.mark.unit
    def test_iz(self, arg_: IZNeuron):
        # verify that all attributes are initialized.
        assert all(hasattr(arg_, attr) for attr in ["identifier", "input", "voltage", "spiked"])

        # test forward pass
        [arg_.forward(data=torch.ones_like(arg_.voltage, device=TEST_DEVICE) * 10.0) for _ in range(STEPS)]


if __name__ == "__main__":
    pytest.main(args=["-s", "-v"])
