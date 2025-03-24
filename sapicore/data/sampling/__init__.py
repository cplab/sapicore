""" Data sampling utility classes.

Leverages scikit-learn and pandas to implement versatile cross validation and sampling schemes.

"""
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator

__all__ = ("BalancedSampler", "CV")


class BalancedSampler:
    """Sampler base class.

    Leverages pandas' group-by operation to parsimoniously perform (dis)proportionate random sampling,
    with or without replacement.

    Parameters
    ----------
    stratified: bool
        By default (`False`), attempts to draw the same number of buffer from each group regardless of imbalances.
        If `True`, uses stratified sampling (proportionate, preserving class imbalances).

    replace: bool
        Whether to sample with replacement. Defaults to `False`.

    Note
    ----
    :meth:`~data.Data.sample` accepts initialized objects of this class.
    Additional samplers may be implemented in future versions.

    """

    def __init__(self, stratified: bool = False, replace: bool = False):
        self.stratified = stratified
        self.replace = replace

    def __call__(self, frame: DataFrame, group_keys: str | list[str], n: int | float):
        """Draws buffer from `frame` after grouping it by `columns`.

        Parameters
        ----------
        frame: DataFrame
            Pandas dataframe containing descriptor labels.

        group_keys: str or list of str
            Names of descriptors (column headers) to group the dataframe by.

        n: int or float
            Specifies the number or fraction of buffer to draw from each group.
            While the former is more convenient for equal (disproportionate) sampling and the latter
            for stratified (proportionate) sampling, integer or float values can be used with either one.

        """
        # keep track of global indices for subset selection.
        frame["index"] = frame.index
        grouped = frame.groupby(group_keys, group_keys=False)

        # convert `n` to integer if need be.
        if isinstance(n, float):
            n = int(n * len(frame["index"].tolist()))

        subset = grouped.apply(lambda x: x.sample(n, replace=self.replace), include_groups=False)

        return subset["index"].tolist()


class CV:
    """Cross validation base class.

    Parameter
    ---------
    data: Data
        The :class:`data.Data` object for which to generate the cross validator.

    cross_validator: BaseCrossValidator
        A scikit-learn :class:`BaseCrossValidator` object.

    label_keys: str or list of str, optional
        One or more descriptor label names. Required for some CV schemes (e.g., for StratifiedKFold, it would be
        the descriptor whose class label counts should be consistent with their proportions in the full sample).

    group_key: str, optional
        A grouping descriptor key. Required for some CV schemes (e.g., GroupKFold, for generating batch-like
        folds with non-overlapping groups w.r.t. some descriptor).

    Note
    ----
    Some external libraries extend the cross validator offerings of scikit-learn.
    This implementation is compatible with any class derived from BaseCrossValidator.

    """

    def __init__(
        self,
        data,
        cross_validator: BaseCrossValidator,
        label_keys: str | list[str] = None,
        group_key: str = None,
        **kwargs
    ):

        self.data = data
        self.cross_validator = cross_validator

        if isinstance(label_keys, str):
            self.labels = self.data.metadata[label_keys][:]

        else:
            # if multiple stratification labels were provided.
            self.labels = list(zip(*[self.data.metadata[i][:] for i in label_keys]))

        self.groups = self.data.metadata[group_key][:] if group_key else None

    def __iter__(self):
        self.index = 0
        splitter = self.cross_validator.split(X=self.labels, y=self.labels, groups=self.groups)

        return splitter

    def __len__(self) -> int:
        return len(self.labels)
