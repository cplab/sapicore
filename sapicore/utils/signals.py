""" Utility classes and functions for artificial signal generation. """
import logging

import torch
from torch import Tensor

from sapicore.utils.constants import DT

__all__ = ("Wave", "WaveIterable", "extend_input_current")


class WaveIterable:
    """Implicit iterator returned by the :class:`Wave` iterable.

    I.e. when you do ``iter(wave)`` or ``for x in wave:...`` it implicitly
    creates an instance of this class and uses it to iterate over.

    Parameters
    ----------
    wave: :class:`Wave:`
        The wave class that will be iterated over.
    """

    def __init__(self, wave):
        self.index = 0
        self.wave = wave

    def __next__(self):
        value = self.wave._next(self.index)
        self.index += 1
        return value


class Wave:
    """Waves are iterables that produce the next value of a specified combination of sine waves, as needed.
    This allows for the generation of arbitrary waveforms with no concern for memory usage.

    Supports basic phase-amplitude coupling as well as phase shifting frequency components with respect to each other.

    Parameters
    ----------
    amplitudes: Tensor
        Amplitudes of each frequency component, in arbitrary units.

    frequencies: Tensor
        Sine component frequencies, in Hz.

    phase_shifts: Tensor
        Starting phase of each frequency component, to be multiplied by np.pi.

    phase_amplitude_coupling: Tensor, optional
        Frequency of variations in amplitude of each component, if such coupling is desired. Defaults to `None`.

    baseline_shift: Tensor, optional
        Value by which to shift the entire signal (on the y-axis).

    sampling_rate: int, optional
        Sampling rate in Hz. Defaults to 1000.0 / :attr:`~utils.constants.DT` (simulation time step).

    device: str, optional
        Specifies a hardware device on which to process tensors, e.g. "cuda:0". Defaults to "cpu".

    See Also
    --------
    :class:`~engine.neuron.analog.oscillator.OscillatorNeuron`:
        An analog neuron that uses a Wave iterator to generate its output.

    Example
    -------
    Define a fast oscillation amplitude-coupled to the phase of a slower oscillation, maxing out at its trough:

        >>> sampling_rate = 1000
        >>> amplitudes = torch.tensor([0.1, 1.0])
        >>> frequencies = torch.tensor([50.0, 5.0])
        >>> phase_shifts = torch.tensor([torch.pi, 0.0])
        >>> phase_amplitude_coupling = torch.tensor([frequencies[1], 0.0])

    Prepare a signal based on the above parameters:

        >>> signal = Wave(amplitudes=amplitudes, frequencies=frequencies, phase_shifts = phase_shifts,
        ... phase_amplitude_coupling=phase_amplitude_coupling, sampling_rate=sampling_rate)

    Generate and plot two seconds of the signal:

        >>> import matplotlib.pyplot as plt
        >>> values = [next(signal) for _ in range(sampling_rate * 2)]
        >>> fig = plt.plot(torch.tensor(values))
        >>> plt.show()

    """

    def __init__(
        self,
        amplitudes: Tensor,
        frequencies: Tensor,
        phase_shifts: Tensor,
        phase_amplitude_coupling: Tensor = None,
        baseline_shift: Tensor = None,
        sampling_rate: float = 1000.0 / DT,
        device: str = "cpu",
    ):
        # instance attributes defining the waveform.
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.phase_shifts = phase_shifts
        self.phase_amplitude_coupling = phase_amplitude_coupling
        self.baseline_shift = baseline_shift
        self.sampling_rate = sampling_rate

        # hardware device on which to store this instance's tensors.
        self.device = device

        # iteration counter.
        self.index = 0

        # validate correctness of waveform configuration provided.
        self._validate()

    def __iter__(self):
        # validate correctness in case params were manually changed
        self._validate()

        return WaveIterable(self)

    def _next(self, index):
        # compute wave at current time point from given components.
        value = torch.zeros(1, dtype=torch.float, device=self.device)

        for i in range(len(self.frequencies)):
            amp_series = self.amplitudes[i]
            time = index * (1.0 / self.sampling_rate)

            if self.phase_amplitude_coupling is not None:
                # add sine variation to base amplitude.
                temp = self.amplitudes[i] * torch.sin(
                    2 * torch.pi * self.phase_amplitude_coupling[i] * time + self.phase_shifts[i]
                )
                amp_series = amp_series + temp

            # add PAC oscillation to combined wave.
            value = value + amp_series * torch.sin(2.0 * torch.pi * self.frequencies[i] * time + self.phase_shifts[i])

        return value + self.baseline_shift

    def _validate(self):
        """Validates wave parameters, setting :attr:`valid` to False if configuration is faulty.
        Specifically, checks that all sine components have an amplitude, a frequency, and a phase
        (i.e., that lengths are equal)."""
        lengths = list(map(len, (self.amplitudes, self.frequencies, self.phase_shifts)))
        if not all(x == lengths[0] for x in lengths):
            raise ValueError("Not all parameters are the same length")


def extend_input_current(
    blueprint: Tensor, num_steps: int, num_units: int, device: str = "cpu", mode: str = "smear"
) -> Tensor:
    """Accepts a tensor of values `blueprint` and either smears or repeats it, depending on `mode`.

    Convenience function for generating a dummy input current from a `blueprint`. That is to say,
    the user provides the desired **shape** of the current time series and specifies how to extend it.

    For instance, if mode is "smear", [0, 1, 0, 2] will be extended to [0, 0, 1, 1, 0, 0, 2, 2, ...]
    to fit the requested duration in simulation step units, `num_steps`.  If mode is "repeat",
    [0, 1, 0, 2] will be extended to [0, 1, 0, 2, 0, 1, 0, 2, ...].

    Parameters
    ----------
    blueprint: Tensor
        Input current values to be extended (typically given under simulation/current in the configuration YAML).

    num_steps: int
        Length of current to be generated, equal to the number of simulation steps.
        This is also the number of columns in the returned tensor.

    num_units: int
        Number of neurons (tensor elements) in the ensemble that will receive the generated current.
        This is also the number of rows in the returned tensor.

    device: str, optional
        Specifies a hardware device on which to register PyTorch buffer fields, e.g. "cuda:0". Defaults to "cpu".

    mode: str, optional
        Whether to "smear" the given blueprint or "repeat" it (as in np.tile).

    Returns
    -------
    Tensor
        The generated current, given as a tensor whose columns are its values at each time point (ensemble elements
        correspond to rows).

    Examples
    --------

        >>> extend_input_current(blueprint=torch.tensor([0, 1, 1, 0]), num_steps=8, num_units=2, mode="smear")
        tensor([[0., 0., 1., 1., 1., 1., 0., 0.],
                [0., 0., 1., 1., 1., 1., 0., 0.]])

        >>> extend_input_current(blueprint=torch.tensor([0, 1, 1, 0]), num_steps=8, num_units=2, mode="repeat")
        tensor([[0., 1., 1., 0., 0., 1., 1., 0.],
                [0., 1., 1., 0., 0., 1., 1., 0.]])

    Note
    ----
    Providing a nonsensical `num_steps` (shorter than or not divisible by the length of the blueprint)
    will result in truncating or extending the output to the nearest integer multiple of the blueprint length.

    Warning
    -------
    The present implementation supports duplication of the current time series to accommodate multiple units.
    :class:`~pipeline.simulation.SimpleSimulator` uses this function to support sending varying dummy currents
    to different ensemble elements.

    Note, however, that these mechanisms are better suited for :mod:`data.synthesis`
    and will be implemented there in a principled manner. This utility function may be deprecated or moved.

    """
    # generate random current data of shape `num_steps` by `num_units`.
    chunk = blueprint.to(device)
    num_parts = num_steps // int(chunk.shape[0])
    part_size = int(chunk.shape[0])

    # sanity check the request.
    if part_size > num_steps:
        logging.warning(
            f"[extend_input_current] Your blueprint is shorter ({part_size}) than the number of steps "
            f"requested ({num_steps}). "
        )

        # use the given blueprint as is.
        num_parts = 1

    elif bool(num_steps % part_size):
        logging.warning(
            f"[extend_input_current] The number of steps you requested ({num_steps}) is not divisible "
            f"by the length of your blueprint ({part_size}). "
        )

    current_evolution = torch.zeros(part_size * num_parts, dtype=torch.float, device=device)

    for i in range(part_size):
        for j in range(num_parts):
            if mode == "smear":
                current_evolution[j + i * num_parts] = chunk[i]
            elif mode == "repeat":
                current_evolution[i + j * part_size] = chunk[i]

    current_evolution = current_evolution.repeat((num_units, 1))
    return current_evolution
