""" Analog ensemble variants.

Currently, :class:`~engine.neuron.analog.AnalogNeuron` or :class:`~engine.neuron.analog.oscillator.OscillatorNeuron`
can be used as base classes. Analog neurons simply add the input to the state stored in `voltage`, while oscillators
are capable of generating an arbitrary combination of sine waves, with basic support for phase-amplitude coupling
via component amplitude oscillations.

Ensembles can be initialized programmatically or from a configuration YAML containing neuron parameters and their
generation method (fixed, zipped, grid, or from distribution). Forward calls invoke the parent neuron class method.

See Also
--------
:class:`~engine.neuron.analog.AnalogNeuron`
:class:`~engine.neuron.analog.oscillator.OscillatorNeuron`

"""
import torch

from sapicore.engine.ensemble import Ensemble
from sapicore.engine.neuron.analog import AnalogNeuron
from sapicore.engine.neuron.analog.oscillator import OscillatorNeuron

__all__ = ("AnalogEnsemble", "OscillatorEnsemble")


class AnalogEnsemble(Ensemble, AnalogNeuron):
    """Ensemble of generic analog neurons."""

    def __init__(self, **kwargs):
        """Constructs an analog ensemble instance, inheriting attributes from
        :class:`~engine.neuron.analog.AnalogNeuron`.

        """
        # invoke parent constructor and initialization method(s).
        super().__init__(**kwargs)


class OscillatorEnsemble(Ensemble, OscillatorNeuron):
    """Ensemble of oscillators.

    Parameters
    ----------
    num_wave_comps: int
        Number of components in each composite sinusoidal.

    Note
    ----
    To initialize an ensemble where the number of frequency components differs across elements,
    set `num_wave_comps` to the maximal number and unnecessary elements to 0.

    For instance, frequencies: [[34.0, 41.0, 50.0], [30.0, 0.0, 0.0]] would yield a 2-ensemble where the first
    element is the sum of three sine components and the second element is a simple 30Hz wave.

    Sapicore does not currently support the configuration syntax [[34.0, 41.0, 50.0], [30.0]] due to the
    inconvenience of dealing with variable-length tensors.

    """

    _extensible_props_: tuple[str] = ("amplitudes", "frequencies", "phases", "amp_freq")

    def __init__(self, num_wave_comps: int = 1, **kwargs):
        """Constructs an oscillator ensemble instance, inheriting attributes from
        :class:`~engine.ensemble.Ensemble` and :class:`~engine.neuron.analog.oscillator.OscillatorNeuron`."""
        # generic ensemble initialization with configurable and loggable tensors.
        super().__init__(**kwargs)

        # number of frequency components each oscillator consists of (use max if variable).
        self.num_wave_comps = num_wave_comps

        # expand extensible tensors to accommodate 2D specification format of multi-component sine waves.
        for prop in self.extensible_props:
            temp = getattr(self, prop)
            zeros = torch.zeros(size=(self.num_units, self.num_wave_comps), dtype=torch.float, device=self.device)

            setattr(self, prop, zeros + temp.reshape(self.num_units, 1))

        # create the wave iterator for each element of this oscillator ensemble.
        self.register_waveforms()

    def heterogenize(self, num_combinations: int, unravel: bool = True):
        """Ensures that calling an oscillator's `heterogenize()` will result in a correct row-wise treatment of
        tensor attributes and update the waveform iterators.

        Note
        ----
        This is necessary, as `heterogenize` is sometimes invoked from ambiguous ensemble classes (e.g., in `Network`,
        when ensembles are initialized from file and dynamically imported).

        """
        super().heterogenize(num_combinations=self.num_units, unravel=False)
        self.register_waveforms()
