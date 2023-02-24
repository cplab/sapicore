""" Utility classes and functions for artificial signal generation. """
import torch
from torch import Tensor

from sapicore.utils.constants import DT

__all__ = ("Wave", "design_input_current", "flatten")


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

    phase_amplitude_coupling: Tensor
        Optional, frequency of variations in amplitude for each component, if coupling is desired.

    sampling_rate: int, optional
        Sampling rate in Hz, by default 1000.0 / DT (simulation time step).

    device: str, optional
        Specifies a hardware device on which to register PyTorch buffer fields, e.g. "cuda:0". Defaults to "cpu".

    See Also
    --------
    neuron.analog.oscillator.OscillatorNeuron:
        An analog neuron that used a Wave iterator to generate its output.

    Example
    -------
    Define a fast oscillation amplitude-coupled to the phase of a slower oscillation, maxing out at its trough:

        >>> sampling_rate = 1000
        >>> amplitudes = torch.tensor([0.1, 1.0])
        >>> frequencies = torch.tensor([50.0, 5.0])
        >>> phase_shifts = torch.tensor([torch.pi, 0.0])
        >>> phase_amplitude_coupling = torch.tensor([frequencies[1], 0.0])

    Initialize a signal based on the above parameters:

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
        sampling_rate: float = 1000.0 / DT,
        device: str = "cpu",
    ):
        # iteration counter.
        self.index = 0

        # instance attributes defining the waveform.
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.phase_shifts = phase_shifts
        self.phase_amplitude_coupling = phase_amplitude_coupling
        self.sampling_rate = sampling_rate

        # hardware device on which to store this instance's tensors.
        self.device = device

        # validate correctness of waveform configuration provided.
        self.valid = self._validate()

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        # compute wave at current time point from given components.
        value = torch.zeros(1, dtype=torch.float, device=self.device)

        if self.valid:
            for i in range(len(self.frequencies)):
                amp_series = self.amplitudes[i]
                time = self.index * (1.0 / self.sampling_rate)

                if self.phase_amplitude_coupling is not None:
                    # add sine variation to base amplitude.
                    temp = self.amplitudes[i] * torch.sin(
                        2 * torch.pi * self.phase_amplitude_coupling[i] * time + self.phase_shifts[i]
                    )
                    amp_series = amp_series + temp

                # add PAC oscillation to combined wave.
                value = value + amp_series * torch.sin(2 * torch.pi * self.frequencies[i] * time + self.phase_shifts[i])

        self.index += 1
        return value

    def _validate(self) -> bool:
        """Validates wave parameters, setting :attr:``valid`` to False if configuration is faulty."""
        # check that all component sines have an amplitude, a frequency, and a phase (i.e., that lengths are equal).
        lengths = [len(v) for v in [self.amplitudes, self.frequencies, self.phase_shifts]]
        if not all(x == lengths[0] for x in lengths):
            return False

        return True


def flatten(lst: list[list]) -> list:
    """Flattens a list."""
    return [item for sublist in lst for item in sublist]


def design_input_current(current: Tensor, num_steps: int, num_units: int, device: str = "cpu") -> Tensor:
    """Convenience function for the quick configuration of a structured ``current``. Accepts a list of float values
    and smears it; e.g., [0, 1, 0, 2] is extended to [0, 0, 1, 1, 0, 0, 2, 2, ...] to fit the requested duration in
    simulation step units, ``num_steps``.

    Parameters
    ----------
    current: Tensor
        Input current values to be smeared (typically given under simulation/current in the configuration YAML).

    num_steps: int
        Length of current to be generated, equal to the number of simulation steps.

    num_units: int
        Number of neurons (tensor elements) in the ensemble that will receive the generated current.

    device: str, optional
        Specifies a hardware device on which to register PyTorch buffer fields, e.g. "cuda:0". Defaults to "cpu".

    Returns
    -------
    Tensor
        The generated current, given as a float32 tensor whose elements are its values at each time point.

    Note
    ----
    The present implementation supports duplication of the current time series to accommodate multiple units.
    Mechanisms for sending varying currents to different ensemble elements are better suited for the data synthesis
    module, and should be implemented there in a principled manner.

    """

    # generate random current data of shape num_steps by num_units.
    chunk = current.to(device)
    num_parts = num_steps // int(chunk.shape[0])
    part_size = int(chunk.shape[0])
    current_evolution = torch.zeros(part_size * num_parts, dtype=torch.float, device=device)

    for i in range(part_size):
        for j in range(num_parts):
            current_evolution[j + i * num_parts] = chunk[i]

    current_evolution = current_evolution.repeat((num_units, 1))
    return current_evolution
