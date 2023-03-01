""" Oscillators are waveform generators.

Oscillators, inheriting from :class:`~engine.neuron.analog.AnalogNeuron`, generate a wave consisting of pure sines.
Components can be phase-shifted relative to each other or have their amplitude coupled to a certain frequency.

Potential uses include the imposition of modulation or forced oscillations. Like other :class:`~engine.neuron.Neuron`
instances, oscillators maintain their numeric state in the tensor `voltage`.

Warning
-------
Oscillators should only emit to non-spiking-dependent synapses (e.g, :class:`~engine.synapse.Synapse`). If paired with
synapses whose behavior requires reading out a `spiked` attribute (e.g., :class:`~engine.synapse.STDP.STDPSynapse`),
the simulation will raise an exception.

See Also
--------
utils.signals.Wave:
    Iterator class used to generate complex waveforms without memory overhead.

"""
from math import pi

import torch
from torch import tensor, Tensor

from sapicore.engine.neuron.analog import AnalogNeuron
from sapicore.utils.signals import Wave

__all__ = ("OscillatorNeuron",)


class OscillatorNeuron(AnalogNeuron):
    """Oscillator neuron implementation.

    Waveform is determined by amplitude, frequency, phase, and coupling configuration parameters in a
    format acceptable to the utility class :class:`~utils.signals.Wave` constructor.

    Parameters
    ----------
    amplitudes: Tensor or list of float
        Component amplitudes in arbitrary units.

    frequencies: Tensor or list of float
        Component frequencies in Hz.

    phases: Tensor or list of float
        Component starting phases between 0 and 2, to be multiplied by pi.

    amp_freq: Tensor or list of float
        Frequencies for amplitude modulation of Nth item, e.g. if frequencies[0] = 40.0
        and amp_freq[0] = 5.0, the amplitude of the 40 Hz component will oscillate at 5 Hz.

    **kwargs:
        Accepts and applies any keyword argument by invoking the parent class :class:`~engine.neuron.Neuron`
        constructor.

    """

    # loggable properties are the same as those of the parent class AnalogNeuron.
    _config_props_: tuple[str] = ("amplitudes", "frequencies", "phases", "amp_freq")

    def __init__(
        self,
        amplitudes: Tensor | list[float] = None,
        frequencies: Tensor | list[float] = None,
        phases: Tensor | list[float] = None,
        amp_freq: Tensor | list[float] = None,
        **kwargs
    ):
        """Constructs an iterator-based oscillator."""
        super().__init__(**kwargs)

        # cast list arguments to tensors if need be.
        amplitudes = tensor(amplitudes, device=self.device) if isinstance(amplitudes, list) else amplitudes
        frequencies = tensor(frequencies, device=self.device) if isinstance(frequencies, list) else frequencies
        phases = tensor(phases, device=self.device) if isinstance(phases, list) else phases
        amp_freq = tensor(amp_freq, device=self.device) if isinstance(amp_freq, list) else amp_freq

        # set configurable oscillator parameters to given tensor or to default if none.
        self.amplitudes = amplitudes if amplitudes is not None else torch.tensor([5.0], device=self.device)
        self.frequencies = frequencies if frequencies is not None else torch.tensor([40.0], device=self.device)
        self.phases = phases if phases is not None else torch.tensor([0.0], device=self.device)
        self.amp_freq = amp_freq if amp_freq is not None else torch.tensor([0.0], device=self.device)

        # placeholder for wave iterator.
        self.iter = []

    def register_waveforms(self) -> None:
        """Creates Wave iterators for each oscillator element."""

        # create an iterator for each oscillator in the ensemble and append it to `iter`.
        self.iter = []
        for i in range(self.amplitudes.shape[0]):
            specification = {
                "amplitudes": self.amplitudes[i],
                "frequencies": self.frequencies[i],
                # convert phase specification from multiples-of-pi to radians.
                "phase_shifts": self.phases[i] * pi,
                "phase_amplitude_coupling": self.amp_freq[i],
            }
            self.iter.append(Wave(**specification, sampling_rate=self.dt * 1000.0, device=self.device))

    def forward(self, data: Tensor) -> dict:
        """Oscillator forward method.

        Advances the wave by one time unit and adds the `data` tensor.

        Parameters
        ----------
        data: Tensor
            External input current to be added to the waveform.

        Returns
        -------
        dict
            Dictionary containing the numeric state tensor `voltage`.

        """
        # compute the next value of the wave and store it in `voltage`, the state variable of the analog neuron.
        # if input current is provided (`data` tensor), it will be added to the oscillation.
        self.voltage = tensor([next(wave_sample) for wave_sample in self.iter], device=self.device).add(data)
        self.simulation_step += 1

        # return current state(s) of loggable attributes as a dictionary.
        return self.state()
