""" Sweep hyperparameter or argument search spaces. """
from typing import Any

import torch
import scipy
import numpy as np

from sapicore.utils.constants import SEED
from sapicore.utils.seed import set_seed

from sklearn.model_selection import ParameterGrid


class Sweep:
    """Reads a `search_space` dictionary and generates appropriate parameter combinations.

    Parameters
    ----------
    search_space: dict
        Argument values, distributions, and sweep strategies. Stored in the `parameters` portion of the
        specification dictionary or YAML.

    Configuration dictionary parameters are expected to be found under one of four keys:

    1. Fixed:

        The argument takes the same value across all execution branches. E.g., if
        settings["fixed"]["resample"]=100, `self.method` will always be called with resample=100.

    2. Zipped:

        The values of the arguments listed are iterated over such that the Nth value of any argument is
        matched with the Nth value of other arguments. Combinations are repeated as many times as it takes
        to reach the number of units dictated by construction, considering all other argument specs.

        E.g., if settings["zipped"]["freq_band"] = [[5, 10], [30, 50]] and ...["resample"] = [50, 100],
        then the procedure referenced by `self.method` will pass freq_band = [5, 10] with resample = 50
        in half of the branches, and freq_band = [30, 50] with resample = 100 in the other half.

    3. Grid:

        The values of the arguments keys listed are iterated over such that any argument value combination is
        equally represented in the execution tree. Combinations are repeated as many times as it takes to reach
        the number of units dictated by the specification dictionary, considering all other argument specs.

        E.g., if settings["zipped"]["freq_band"] = [[5, 10], [30, 50]] and ...["resample"] = [50, 100],
        then the procedure referenced by `self.method` will be called in four qualitatively distinct branches,
        where (freq_band, resample) take the values ([5, 10], 50), ([5, 10], 100), ([30, 50], 50), ([30, 50], 100).

    4. Random:

        The values of the parameter keys listed are drawn from a :mod:`scipy.stats` distribution given under the
        key "method", with the arguments "args". E.g., if settings["random"]["resample"]["method"]="uniform"
        and ...["args"] = [x, y], then `resample` values will be drawn from `U(x, x+y)`.

    Example
    -------
    Define a search space and generate a list of parameter combination dictionaries:

        >>> s = Sweep(search_space={"fixed": {"a": 4}, "grid": {"b": [6, 7], "c": [8, 9]}}, num_combinations=4)
        >>> print(s.combinations)
        [{'a': 4, 'b': 6, 'c': 8}, {'a': 4, 'b': 6, 'c': 9}, {'a': 4, 'b': 7, 'c': 8}, {'a': 4, 'b': 7, 'c': 9}]

    See Also
    --------
    `Scipy distributions <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`_
        For a list of supported probability distributions and their arguments.

    """

    def __init__(self, search_space: dict, num_combinations: int = None):
        # set random seed for consistency across runs.
        set_seed(SEED)

        self.search_space = search_space
        self.num_combinations = num_combinations

        # derive the argument combinations defining the execution branches from `search_space`.
        self.combinations = self.set_combinations()

    def __call__(self, *args, **kwargs):
        return self.combinations

    def heterogenize(self, obj: Any, unravel: bool = True):
        """Initializes `obj`'s attribute tensors heterogeneously based on a given sweep search dictionary.

        Parameters
        ----------
        obj: Component
            Sapicore engine component whose parameters will be diversified (e.g., neuron, ensemble, synapse).

        unravel: bool
            When the property is a 2D tensor, dictates whether combination values should be assigned
            to individual elements (True) or overwrite entire rows (False).

        """
        # for every parameter combination, set the appropriate attribute tensor element values.
        for i, combination in enumerate(self.combinations):
            for key, value in combination.items():
                # cast attribute value to tensor on the correct device.
                value = torch.tensor(value, device=obj.device)

                # assumes attribute to be modified is already a tensor after super().__init__.
                if unravel:
                    # compute unraveled index given the shape of the tensor to be heterogenized.
                    unraveled_index = np.unravel_index(i, getattr(obj, key).shape)
                    getattr(obj, key)[unraveled_index] = value

                else:
                    # handle the case where values should overwrite attribute tensor rows.
                    getattr(obj, key)[i] = value

    def set_combinations(self) -> list[dict]:
        """Determine the argument combinations defining this sweep and update the `combinations` attribute."""
        # allow user to set an arbitrary number of combinations, e.g. in case of random sampling.
        # if not specified, `count_combinations` will use the LCM of grid and zipped combinations to set the total.
        self.count_combinations()

        # initialize an empty list to contain kwargs-like dictionaries, one for each branch/combination.
        combinations = []

        # combinations to cycle through when setting zipped argument values,
        zipped_cycle = self.extract_combinations(self.search_space, mode="zipped")

        # combinations to cycle through when setting grid search argument values.
        grid_cycle = self.extract_combinations(self.search_space, mode="grid")

        for n in range(self.num_combinations):
            # initialize temporary array buffer to store combination dictionary from this iteration.
            combinations.append({})

            # handle the fixed parameters first.
            if "fixed" in self.search_space:
                for arg in self.search_space["fixed"].keys():
                    combinations[n][arg] = self.search_space["fixed"][arg]

            # handle the zipped parameters next, interleaving configurations.
            if "zipped" in self.search_space:
                for arg in self.search_space["zipped"].keys():
                    combinations[n][arg] = zipped_cycle[int(n % len(zipped_cycle))][arg]

            # handle the grid-search parameters next, interleaving configurations.
            if "grid" in self.search_space:
                for arg in self.search_space["grid"].keys():
                    combinations[n][arg] = grid_cycle[int(n % len(grid_cycle))][arg]

        # finally, draw random parameter values from their scipy distributions.
        if "random" in self.search_space:
            for arg in self.search_space["random"].keys():
                random_distribution = getattr(scipy.stats, self.search_space["random"][arg]["method"])
                random_variable = random_distribution(*self.search_space["random"][arg]["args"])
                random_values = random_variable.rvs(size=self.num_combinations)

                for n in range(self.num_combinations):
                    combinations[n][arg] = random_values[n]

        return combinations

    def count_combinations(self):
        """Counts search space configurations and sets `num_combinations` if not provided or invalid.

        Enforces the following contract w.r.t. the calculated number of combinations (NoC):

        1. If NoC was not provided by the user, the calculated value is the minimal one ensuring that every unique
        parameter combination can be represented.

        2. If NoC was provided, the calculated value will be either equal to or greater than the requested number.
        The calculated value will be greater iff the user's value will result in an imbalance (i.e., if not
        every parameter combination can be represented with that NoC).

        """
        zipped_combinations = 0
        grid_combinations = 0

        for key, value in self.search_space.items():
            if key == "zipped" and self.search_space["zipped"]:
                # if a nonempty "zipped" strategy exists in the configuration, we have as many additional
                # branches to add as the common length of the zipped parameter options.
                zipped_combinations = len(list(self.search_space["zipped"].values())[0])

            if key == "grid" and self.search_space["grid"]:
                # use the scikit-learn utility class ParameterGrid to extract possible combinations.
                grid_combinations = len(list(ParameterGrid(self.search_space["grid"])))

        # determine number of unique combinations if one was not requested.
        lcm = np.lcm(zipped_combinations, grid_combinations)
        if not self.num_combinations:
            if zipped_combinations + grid_combinations == 0:
                # if we got here without zipped and grid combos, this means user only had fixed variables,
                # or had random variables but didn't set `num_combinations`. In that case, default to 1.
                self.num_combinations = 1

            elif lcm == 0:
                # if only zipped or only grid variables, whichever is nonzero is the number of combinations.
                self.num_combinations = zipped_combinations if zipped_combinations else grid_combinations

        elif self.num_combinations < lcm:
            # if `num_combinations` provided is smaller than the number of unique combinations,
            # set it to the least common multiplier to ensure balanced output.
            self.num_combinations = lcm

    @staticmethod
    def extract_combinations(parameters: dict, mode: str) -> list[dict]:
        """Extracts all parameter combinations from `parameters` based on the traversal `mode` given.

        Parameters
        ----------
        parameters: dict
            Dictionary, possibly loaded from YAML, containing parameters and their levels.

        mode: str
            Combination strategy, "grid" if all parameter permutations should be tested or "zipped" for a linear pass.

        See Also
        --------
        :class:`~sweep.Sweep`:
            For a full documentation of parameter sweep strategies.

        """
        combinations = []

        # return list of dictionaries with all possible parameter permutations.
        if mode == "grid":
            if "grid" in parameters:
                parameters = parameters["grid"]
                param_grid = ParameterGrid(parameters)

                for dict_ in param_grid:
                    combinations.append(dict_)

        # return list of dictionaries with serially generated parameter permutations (key_1[n] goes with key_2[n]).
        elif mode == "zipped":
            if "zipped" in parameters:
                parameters = parameters["zipped"]

                keys = list(parameters)

                # verify that, in zipped mode, all parameters have the same number of levels.
                if len(keys) > 1:
                    if not [len(parameters[keys[i]]) == len(parameters[keys[i + 1]]) for i in range(len(keys) - 1)]:
                        raise AssertionError("Different number of levels for one or more parameters.")

                for i in range(len(parameters[keys[0]])):
                    # make copy of full dictionary and remove all levels except i.
                    combinations.append(parameters.copy())
                    for j in keys:
                        combinations[i][j] = combinations[i][j][i]

        else:
            raise ValueError("Invalid parameter combination mode provided.")

        return combinations
