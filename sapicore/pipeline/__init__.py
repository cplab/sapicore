"""Configurable workflows and scripts.

Pipelines are flexible objects consisting of two logical parts: configurable parameters and a custom
script defined by overriding the abstract method :meth:`~Pipeline.run`.

To facilitate the development and testing of models, Sapicore provides a compact default simulation
pipeline with a sample configuration YAML. Advanced users may use those as a basis for custom workflows.

"""
import os
from tree_config import Configurable, load_config, apply_config

__all__ = ("Pipeline",)


class Pipeline(Configurable):
    """Pipeline base class.

    Parameters
    ----------
    configuration: dict or str, optional
        Dictionary or path to YAML to use with :meth:`apply_config`.

    """

    def __init__(self, configuration: dict | str = None, **kwargs):
        # parse configuration from YAML if given and use resulting dictionary to initialize attributes.
        if isinstance(configuration, str) and os.path.exists(configuration):
            apply_config(self, load_config(None, configuration))
            self.configuration = load_config(self, filename=configuration)

        elif isinstance(configuration, dict):
            apply_config(self, configuration)

        # developer may define arbitrary attributes at instantiation.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        """String representation of this pipeline's attributes."""
        return str(vars(self))

    def run(self) -> None:
        """Pipeline implementation, to be overridden by the user."""
        raise NotImplementedError
