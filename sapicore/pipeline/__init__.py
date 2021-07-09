"""Pipeline
===========

A pipeline is a class that can run the training and testing of a model.
It should handle command line input, model configuration, and training and
testing.
"""

__all__ = ('PipelineBase', )


class PipelineBase:
    """Baseclass for pipelines.
    """

    def run(self) -> None:
        """Runs the pipeline.

        To be overwritten by users.
        """
        raise NotImplementedError
