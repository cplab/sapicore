Usage guide
===========

Parameters
----------

Both buffers and parameters are saved and restored with the state.
Parameters are used by the optimizer and returned in the parameters list.
We don't use one so we only use buffers in the project.
We can also use the ``self.register_buffer`` syntax for registering it.
