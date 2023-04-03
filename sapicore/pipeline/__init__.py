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

    Raises
    ------
    FileNotFoundError
        If provided with an invalid string `configuration` path.

    """

    def __init__(self, configuration: dict | str = None, **kwargs):
        if isinstance(configuration, str):
            # parse configuration from YAML if given and use resulting dictionary to initialize attributes.
            content = load_config(None, configuration) if os.path.exists(configuration) else None

            if not content:
                raise FileNotFoundError(f"Could not find a configuration YAML at {configuration}")

            else:
                # apply the configuration to this pipeline object.
                configuration = content
                apply_config(self, configuration)

        self.configuration = configuration

    def run(self) -> None:
        """Pipeline implementation, to be overridden by the user."""
        raise NotImplementedError
