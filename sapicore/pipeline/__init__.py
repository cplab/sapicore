"""Configurable workflows and scripts.

Pipelines are flexible objects consisting of two logical parts: configurable parameters and a custom
script defined by overriding the abstract method :meth:`~Pipeline.run`.

To facilitate the development and testing of models, Sapicore provides a compact default simulation
pipeline with a sample configuration YAML. Advanced users may use those as a basis for custom workflows.

"""
from tree_config import Configurable
from sapicore.utils.io import load_apply_config

__all__ = ("Pipeline",)


class Pipeline(Configurable):
    """Pipeline base class.

    Parameters
    ----------
    config_or_path: dict or str, optional
        Dictionary or path to YAML to use with :meth:`apply_config`.

    Raises
    ------
    FileNotFoundError
        If provided with an invalid string `configuration` path.

    """

    def __init__(self, config_or_path: dict | str = None, **kwargs):
        if isinstance(config_or_path, str):
            config_or_path = load_apply_config(config_or_path, apply_to=self)

        if isinstance(config_or_path, dict):
            # reached whether it was transformed to a dictionary in the conditional above or provided as such.
            self.configuration = config_or_path

    def run(self) -> None:
        """Pipeline implementation, to be overridden by the user."""
        raise NotImplementedError
