""" Data sampling.

Utilize scikit-learn convenience methods to implement versatile cross validation schemes.

"""
from sklearn.model_selection import BaseCrossValidator
from sapicore.data import Data

__all__ = ("CV",)


class CV:
    """Cross validation base class.

    Accepts a scikit-learn cross validator object and maintains an iterator.

    """

    def __init__(self, data: Data, cross_validator: BaseCrossValidator, label_key: str, group_key: str = None):
        self.data = data
        self.cross_validator = cross_validator

        self.labels = self.data.descriptors[label_key].labels
        self.groups = self.data.descriptors[group_key].labels if group_key else None

    def __iter__(self):
        self.index = 0
        self.splitter = self.cross_validator.split(X=self.labels, y=self.labels, groups=self.groups)

        return self.splitter

    def __next__(self):
        self.index += 1
        if self.index < len(self.labels):
            tr, te = next(self.splitter)
            return tr, te
        else:
            raise StopIteration

    def __len__(self) -> int:
        """Returns length of the relevant label list."""
        return len(self.labels)
