""" Utility methods for generating diagnostic plots from loggable variables. """
from torch import Tensor

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

import matplotlib.pyplot as plt


__all__ = ("spike_raster",)


def spike_raster(
    data: Tensor, line_size: float = 0.25, events: ArrayLike | Tensor = None, event_colors: ArrayLike | Tensor = None
) -> plt.eventplot:
    """Generates a spike raster event plot from `data`.

    Parameters
    ----------
    data: Tensor
        2D binary integer tensor containing spike time series in the format units X steps.

    line_size: float
        Controls spike event line size.

    events: ArrayLike or Tensor
        Event timestamps w.r.t. X-axis. The raster will contain semi-transparent vertical bars at those locations.

    event_colors: ArrayLike or Tensor
        Colors to use with vertical lines corresponding to event timestamps, given as a list where
        the i-th element is the color of the i-th vertical line.

    """
    if data.max() <= 1:
        # we are dealing with binary spike data and need to transform it to spike times.
        # assume shape is neurons X time.
        df = pd.DataFrame(data)

        # create series of indices containing positions for raster plot.
        positions = df.apply(lambda x: df.index[x == 1.0])

        # pandas index returns inconsistent output depending on data frame content (e.g., if all rows are identical,
        # returns frame instead of list[Series]), and `plt.eventplot()` is very particular about 2D input format.
        if type(positions) is not pd.Series:
            positions = [i for i in positions.items()]
            positions = np.array(positions, dtype=object)[:, 1]

        else:
            # exit if `positions` is an empty series, as there is no need to plot an empty raster.
            if positions.empty:
                return None

        # Create raster plot with inverted y-axis to display columns in ascending order.
        plot = plt.eventplot(np.transpose(np.array(positions)[:]), linelengths=line_size, colors="black")
        # plt.yticks(range(data.shape[1]))
        plt.yticks([])

        # create vertical bars to represent events, if applicable.
        if events is not None:
            colors = [0] * len(events) if event_colors is None else event_colors
            for i, event in enumerate(events):
                plt.axvline(x=event, color=colors[i], alpha=0.5)

        plt.gca().invert_yaxis()

    else:
        plot = plt.eventplot(data, linelengths=line_size)

    # decorate plot and set axis limits.
    plt.title("Ensemble Spiking Activity")

    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.xlim(0, data.shape[0])

    return plot
