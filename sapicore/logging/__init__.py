"""Logging
==========

:mod:`~sapicore.logging` provides the tools to configure the logging of
Sapicore models. It supports logging of simple properties to tensorboard as
well as arbitrary properties, including tensors and arrays to a H5 file.
Individual properties can be included or excluded from logging using a config
dict that can be saved to a yaml file, similarly to :mod:`tree-config`
configuration.

Loggable
--------

The main API is the :class:`~sapicore.Loggable` that defines an API for
properties to support opting-in to logging. One lists all the properties that
can potentially be logged in :attr:`~sapicore.Loggable._loggable_props_`, on
a per class basis. Sub classing extends
:attr:`~sapicore.Loggable._loggable_props_`, and to get the list of potentially
loggable properties it accumulates all the properties listed in all the
:attr:`~sapicore.Loggable._loggable_props_` of all the super classes in
:attr:`~sapicore.Loggable.loggable_props`.

To supper logging of objects nested in other objects, we use
:attr:`~sapicore.Loggable._loggable_children_` to list all the properties that
are objects that should be inspected for loggable properties.
These are similarly all listed in :attr:`~sapicore.Loggable.loggable_children`.

Configuration
-------------

:func:`read_loggable_from_object`, :func:`read_loggable_from_file`,
:func:`update_loggable_from_object`, :func:`dump_loggable`,
:func:`load_save_get_loggable_properties`, and
:func:`get_loggable_properties` provide the functions to get all the loggable
properties either as a yaml file or dict that can be edited, or as a flat list
of the properties. They all
provide a mapping from the property name to a bool indicting whether the
property should be logged.

E.g. a model such as:

.. code-block:: python

    class Model(Loggable):

        _loggable_props_ = ('name', )

        name = 'chair'

would generate the dict ``{'name': True}`` using
``read_loggable_from_object(Model(), True)`` and a yaml file containing
``name: true`` when dumped using
``dump_config('loggable.yaml', read_loggable_from_object(Model(), True))``.

Logging
-------

:func:`log_tensor_board` supports taking the flat list of properties and
logging all the properties that are set to be logged to the tensorboard
format on disk using a ``torch.utils.tensorboard.SummaryWriter`` instance.

Similarly, :class:`NixLogWriter` supports logging arbitrary tensors and numpy
arrays to a nix HDF5 file. :class:`NixLogReader` can then be used to load the
data from the nix file into scalars or numpy arrays of the original shape.
"""
from typing import Tuple, List, Dict, Any, Union, Optional
from pathlib import Path
import os
import numpy as np
import torch
import nixio as nix
import datetime

import sapicore
from tree_config.utils import get_class_bases
from tree_config.yaml import yaml_dumps, yaml_loads
from tree_config import dump_config, read_config_from_file

__all__ = (
    'Loggable', 'read_loggable_from_object', 'get_loggable_properties',
    'read_loggable_from_file', 'update_loggable_from_object', 'dump_loggable',
    'log_tensor_board', 'load_save_get_loggable_properties', 'NixLogWriter',
    'NixLogReader')

FlatLoggableProps = List[Tuple[List[str], Any, str, bool]]
LogDataItem = Tuple[
    Any, str, nix.DataArray, nix.DataArray, nix.DataArray, nix.DataArray]


class Loggable:
    """The :class:`Loggable` can be used as a base-class for objects that need
    to control which of its properties can be logged. E.g. when getting
    all the properties that need to be logged, :func:`read_loggable_from_object`
    will call :meth:`loggable_props` and :meth:`loggable_children`, which looks
    up all the properties from :attr:`_loggable_props_` and
    :attr:`_loggable_children_`.

    For example:

    .. code-block:: python

        class LoggableModel(Loggable):

            _loggable_props_ = ('frame', 'color')

            frame = 'square'
            color = 'blue'

    Then:

    .. code-block:: python

        >>> read_loggable_from_object(LoggableModel(), True)
        {'frame': True, 'color': True}
    """

    _loggable_props_: Tuple[str] = ()

    _loggable_props: List[str] = None

    _loggable_children_: Dict[str, str] = {}

    _loggable_children: Dict[str, str] = None

    @property
    def loggable_props(self) -> List[str]:
        """A property containing the list of the names of all the properties of
        the instance that could be loggable (i.e. listed in
        ``_loggable_props_`` of the class and it's super classes).

        E.g.:

        .. code-block:: python

            class Model(Loggable):

                _loggable_props_ = ('name', )

                name = 'chair'

        then:

        .. code-block:: python

            >>> model = Model()
            >>> app.loggable_props
            ['name']
        """
        props = self._loggable_props
        if props is None:
            props = {}
            cls = self.__class__

            for c in [cls] + list(get_class_bases(cls)):
                if '_loggable_props_' not in c.__dict__:
                    continue

                for prop in c._loggable_props_:
                    if prop in props:
                        continue

                    if not hasattr(self, prop):
                        raise Exception('Missing attribute <{}> in <{}>'.
                                        format(prop, cls.__name__))
                    props[prop] = None

            props = self._loggable_props = list(props)
        return props

    @property
    def loggable_children(self) -> Dict[str, str]:
        """A property containing the dict of the friendly/property names of
        all the children objects of this instance that could be loggable (i.e.
        listed in ``_loggable_children_`` of the class and it's super classes).

    E.g.:

    .. code-block:: python

        class Model(Loggable):

            _loggable_children_ = {'the box': 'box'}

            box = None

        class Box(Loggable):
            pass

    then:

    .. code-block:: python

        >>> model = Model()
        >>> model.box = Box()
        >>> model.loggable_children
        {'the box': 'box'}
        """
        children = self._loggable_children
        if children is None:
            self._loggable_children = children = {}
            cls = self.__class__

            for c in [cls] + list(get_class_bases(cls)):
                if '_loggable_children_' not in c.__dict__:
                    continue

                for name, prop in c._loggable_children_.items():
                    if name in children:
                        continue

                    if not hasattr(self, prop):
                        raise Exception('Missing attribute <{}> in <{}>'.
                                        format(prop, cls.__name__))
                    children[name] = prop

        return children


def _fill_loggable_from_declared_objects(
        classes: Dict[str, Any], loggable: Dict[str, Any],
        default_value: bool) -> None:
    for name, obj in classes.items():
        if isinstance(obj, dict):
            loggable[name] = {
                k: read_loggable_from_object(o, default_value)
                for k, o in obj.items()}
        elif isinstance(obj, (list, tuple)):
            loggable[name] = [
                read_loggable_from_object(o, default_value) for o in obj]
        else:
            loggable[name] = read_loggable_from_object(obj, default_value)


def read_loggable_from_object(
        obj: Loggable, default_value: bool) -> Dict[str, Any]:
    """Returns a recursive dict containing all the loggable properties of the
    obj and its loggable children. Each item maps the item name to
    ``default_value``, indicating whether to log the item by default.

    :param obj: The object from which to get the loggables.
    :param default_value: A bool (True/False), indicating whether to default
        the loggable properties to True (they are logged) or False (they are
        not logged). The individual properties can subsequently be set either
        way manually in the dict.

    E.g.:

    .. code-block:: python

        class Model(Loggable):

            _loggable_props_ = ('name', )

            _loggable_children_ = {'the box': 'box'}

            name = 'chair'

            box = None

        class Box(Loggable):

            _loggable_props_ = ('volume', )

            volume = 12

    then:

    .. code-block:: python

        >>> model = Model()
        >>> model.box = Box()
        >>> loggables = read_loggable_from_object(model, False)
        >>> loggables
        {'the box': {'volume': False}, 'name': False'}
        >>> # to log name
        >>> loggables['name'] = True
    """
    # TODO: break infinite cycle if obj is listed in its nested loggable
    #  classes
    loggable = {}

    # get all the loggable classes used by the obj
    objects = {
        name: getattr(obj, prop)
        for name, prop in obj.loggable_children.items()}

    _fill_loggable_from_declared_objects(objects, loggable, default_value)

    for attr in obj.loggable_props:
        if attr not in loggable:
            loggable[attr] = default_value
    return loggable


def _get_loggable_from_declared_objects(
        property_path: List[str], classes: Dict[str, Any],
        friendly_names: Dict[str, str], loggable: Dict[str, Any]
) -> FlatLoggableProps:
    loggable_properties = []
    for name, obj in classes.items():
        if obj is None:
            continue

        friendly_name = friendly_names.get(name, name)
        if friendly_name in loggable:
            loggable_properties.extend(
                get_loggable_properties(
                    obj, loggable[friendly_name], property_path + [name]))

    return loggable_properties


def get_loggable_properties(
        obj: Loggable, loggable: Dict[str, Any],
        property_path: Optional[List[str]] = None
) -> FlatLoggableProps:
    """Takes the loggable data read with :func:`read_loggable_from_object`
    or :func:`read_loggable_from_file`, filters those properties that should
    be logged (i.e. those that are True) and flattens them into a single flat
    list.

    This list could be used to efficiently log all the properties that need
    to be logged, by iterating the list.

    :param obj: The object from which to get the loggables.
    :param loggable: The loggable data previously read with
        :func:`read_loggable_from_object` or :func:`read_loggable_from_file`.
    :param property_path: A list of strings, indicating the object children
        names, starting from some root object that lead to the ``obj``.
        This can be used to reconstruct the full object path to each property
        starting from ``obj``.

        Should just be None (default) in user code.
    :returns: A list of 4-tuples, each containing
        ``(property_path, item, prop, value)``. ``property_path`` is the list
        of strings, indicating the object children names,
        starting from some root object that lead to the ``item``. ``item`` is
        the object or child whose property ``prop`` is being considered.
        ``value`` is a bool indicating whether to log the property.

    E.g.:

    .. code-block:: python

        class Model(Loggable):

            _loggable_props_ = ('name', )

            _loggable_children_ = {'the box': 'box'}

            name = 'chair'

            box = None

        class Box(Loggable):

            _loggable_props_ = ('volume', )

            volume = 12

    then:

    .. code-block:: python

        >>> model = Model()
        >>> model.box = Box()
        >>> d = read_loggable_from_object(model, False)
        {'the box': {'volume': False}, 'name': False'}
        >>> d['the box']['volume'] = True
        >>> get_loggable_properties(model, d)
        [(['the box', 'volume'], <Box at 0x16f0b2b5320>, 'volume', True)]
    """
    property_path = property_path or []
    loggable_properties = []
    # get all the loggable classes used by the obj
    objects = {
        prop: getattr(obj, prop)
        for name, prop in obj.loggable_children.items()}
    friendly_names = {v: k for k, v in obj.loggable_children.items()}
    props = set(obj.loggable_props)

    loggable_properties.extend(
        _get_loggable_from_declared_objects(
            property_path, objects, friendly_names, loggable))

    loggable_values = {
        k: v for k, v in loggable.items() if k not in objects and k in props}

    for k, v in loggable_values.items():
        if v:
            loggable_properties.append((property_path + [k], obj, k, v))

    return loggable_properties


def _update_loggable_from_declared_objects(
        classes: Dict[str, Any], loggable: Dict[str, Any],
        new_loggable: Dict[str, Any], default_value: bool) -> None:
    for name, obj in classes.items():
        if isinstance(obj, dict):
            obj_loggable = loggable.get(name, {})
            new_loggable[name] = {
                k: update_loggable_from_object(
                    o, obj_loggable.get(k, {}), default_value)
                for k, o in obj.items()}
        elif isinstance(obj, (list, tuple)):
            obj_loggable = loggable.get(name, [])
            new_loggable[name] = items = []
            for i, o in enumerate(obj):
                if i < len(obj_loggable):
                    items.append(
                        update_loggable_from_object(
                            o, obj_loggable[i], default_value))
                else:
                    items.append(
                        update_loggable_from_object(o, {}, default_value))
        else:
            new_loggable[name] = update_loggable_from_object(
                obj, loggable.get(name, {}), default_value)


def update_loggable_from_object(
        obj: Loggable, loggable: Dict[str, Any], default_value: bool
) -> Dict[str, Any]:
    """Takes the loggable data read with :func:`read_loggable_from_object`
    or :func:`read_loggable_from_file` and creates a new loggable dict from it
    and updates it by adding any new loggable properties of the object, that has
    been added to the object or its children since the last time it was
    created.

    :param obj: The object from which to get the loggables.
    :param loggable: The the loggable data read with
        :func:`read_loggable_from_object` or :func:`read_loggable_from_file`.
    :param default_value: A bool (True/False), indicating whether to default
        the new loggable properties to True (they are logged) or False (they are
        not logged). The individual properties can subsequently be set either
        way manually in the dict.
    :returns: The updated loggable dict.
    """
    # TODO: break infinite cycle if obj is listed in its nested loggable
    #  classes
    new_loggable = {}

    # get all the loggable classes used by the obj
    objects = {
        name: getattr(obj, prop)
        for name, prop in obj.loggable_children.items()}

    _update_loggable_from_declared_objects(
        objects, loggable, new_loggable, default_value)

    for attr in obj.loggable_props:
        if attr not in new_loggable:
            new_loggable[attr] = loggable.get(attr, default_value)
    return new_loggable


def read_loggable_from_file(filename: Union[str, Path]) -> Dict[str, Any]:
    """Reads and returns the dict indicating whether to log objects from a file
    that was previously dumped with :func:`dump_loggable`.

    :param filename: The filename to the yaml file.

    E.g.:

    .. code-block:: python

        >>> class Model(Loggable):
        >>>     _loggable_props_ = ('name', )
        >>>     name = 'chair'
        >>> model = Model()
        >>> d = read_loggable_from_object(model, True)
        >>> dump_loggable('loggable.yaml', d)
        >>> read_loggable_from_file('loggable.yaml')
        {'name': True}
    """
    return read_config_from_file(filename)


def dump_loggable(filename: Union[str, Path], data: Dict[str, Any]) -> None:
    """Dumps the loggable data gotten with e.g.
    :func:`read_loggable_from_object` to a yaml file.

    :param filename: The yaml filename.
    :param data: The loggable data.

    E.g.:

    .. code-block:: python

        >>> class Model(Loggable):
        >>>     _config_props_ = ('name', )
        >>>     name = 'chair'
        >>> model = Model()
        >>> d = read_loggable_from_object(model, True)
        >>> dump_config('loggable.yaml', d)
        >>> with open('loggable.yaml') as fh:
        ...     print(fh.read())

    Which prints ``name: true``.
    """
    dump_config(filename, data)


def log_tensor_board(
        writer, loggable_properties: FlatLoggableProps,
        global_step: Optional[int] = None, walltime: Optional[float] = None,
        prefix: str = '') -> None:
    """Logs the current value of all the given properties to the TensorBoard
    writer object for live display.

    Properties that are arrays are logged as a sequence of scalars, which is
    inefficient. It is suggested to use :class:`NixLogWriter` for complex
    data and :func:`log_tensor_board` only for scalars.

    :param writer: An instantiated ``torch.utils.tensorboard.SummaryWriter``
        object to which the data will be written.
    :param loggable_properties: The list of properties (in the format of
        :func:`get_loggable_properties`) to read and log.
    :param global_step: The optional global timestep, which must be
        monotonically increasing if providing.
    :param walltime: The optional wall time.
    :param prefix: An optional prefix used to log the data under.
    """
    for names, obj, prop, selection in loggable_properties:
        if not selection:
            continue

        value = getattr(obj, prop)
        if prefix:
            names = [prefix] + list(names)
        tag = '/'.join(names)

        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        else:
            value = np.asarray(value)

        if value.shape:
            if isinstance(selection, (list, tuple)):
                flat_idx = np.ravel_multi_index(
                    np.array(selection).T, value.shape)
                flat_val = np.take(np.ravel(value), flat_idx)
                items = {
                    ','.join(map(str, idx)): val for idx, val in zip(
                        selection, flat_val)}
            else:
                items = {','.join(map(str, idx)): val
                         for idx, val in np.ndenumerate(value)}
            writer.add_scalars(
                tag, items, global_step=global_step, walltime=walltime)
        else:
            writer.add_scalar(
                tag, value.item(), global_step=global_step, walltime=walltime)


def load_save_get_loggable_properties(
        obj: Any, filename: Union[str, Path], default_value: bool
) -> FlatLoggableProps:
    """Loads the loggable configuration from the yaml file (
    :func:`read_loggable_from_file`, if the file doesn't
    exist it creates it with :func:`read_loggable_from_object` and
    :func:`dump_loggable`), updates it with :func:`update_loggable_from_object`
    and then dumps it back to the yaml file with :func:`dump_loggable`. It also
    returns the flattened loggable properties using
    :func:`get_loggable_properties`.

    This can be used to get the loggable properties, but also making sure the
    file contains the current loggables including any new properties not
    previously there.

    :param obj: The loggable object.
    :param filename: The yaml filename.
    :param default_value: A bool (True/False), indicating whether to default
        the new loggable properties to True (they are logged) or False (they are
        not logged). The individual properties can subsequently be set either
        way manually in the file.
    :returns: The loggable list returned by :func:`get_loggable_properties`.

    E.x.:

    .. code-block:: python

        class Model(Loggable):

            _config_props_ = ('name', )

            name = 'chair'

        class ModelV2(Loggable):

            _config_props_ = ('name', 'side')

            name = 'chair'

            side = 'left'

    then:

    .. code-block:: python

        >>> model = Model()
        >>> load_save_get_loggable_properties(model, 'loggable.yaml', True)
        [(['name'], <Model at 0x16f0b2b5278>, 'name', True)]
        >>> # then later for v2 of the model
        >>> model_v2 = ModelV2()
        >>> load_save_get_loggable_properties(model_v2, 'loggable.yaml', True)
        [(['name'], <ModelV2 at 0x16f0b2b5695>, 'name', True),
         (['side'], <ModelV2 at 0x16f0b2b5695>, 'side', True)]
        >>> with open('loggable.yaml') as fh:
        ...     print(fh.read())

    this prints::

        name: true
        side: true
    """
    if not os.path.exists(filename):
        dump_loggable(filename, read_loggable_from_object(obj, default_value))
    loggable = read_loggable_from_file(filename)
    dump_loggable(
        filename, update_loggable_from_object(obj, loggable, default_value))

    return get_loggable_properties(obj, loggable)


class NixLogWriter:
    """Writer that supports logging properties to a Nix HDF5 file.

    It supports logging arbitrary torch tensors, numpy arrays, and numerical
    scalars. The logged data can then be retrieved with :class:`NixLogReader`.

    A typical example is:

    . code-block:: python

        log_props = read_loggable_from_object(Model(), True)
        writer = NixLogWriter(h5_filename)
        # create the file
        writer.create_file()
        # create a section for this session
        writer.create_block('example')
        # create the objects required for logging using the loggable properties
        property_arrays = writer.get_property_arrays('example', log_props)
        do_work...
        # all the enabled loggable props will be logged
        writer.log(property_arrays, 0)
        do_more_work...
        writer.log(property_arrays, 1)
        # now close
        writer.close_file()
    """

    nix_file: Optional[nix.File] = None
    """The internal nix file object."""

    filename: Union[str, Path] = ''
    """The filename of the nix file that will be created when
    :meth:`create_file` is called.
    """

    git_hash = ''
    """An optional git hash that can be included in the file. It can be used
    to label the model version for example.
    """

    compression = nix.Compression.Auto
    """The compression to use in the file.

    It is one of ``nix.Compression``.
    """

    metadata: Optional[dict] = None
    """A dict of arbitrary metadata that can be included in the file.

    It is saved to the file after being encoded using
    :meth:`~tree_config.yaml.yaml_dumps`.
    """

    config_data: Optional[dict] = None
    """A dict of :mod:`tree_config` config data used to config the model that
    can be included in the file.

    It is saved to the file after being encoded using
    :meth:`~tree_config.yaml.yaml_dumps`.
    """

    def __init__(
            self, filename: Union[str, Path], git_hash='',
            compression=nix.Compression.Auto, metadata: Optional[dict] = None,
            config_data: Optional[dict] = None):
        self.filename = filename
        self.git_hash = git_hash
        self.compression = compression
        self.metadata = metadata
        self.config_data = config_data

    def __enter__(self):
        self.create_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_file()

    def create_file(self) -> None:
        """Creates the file named in :attr:`filename`. If it exists, an error
        is raised.

        :meth:`close_file` must be called after this to close the file.
        """
        if os.path.exists(self.filename):
            raise ValueError(f'{self.filename} already exists')

        self.nix_file = f = nix.File.open(
            str(self.filename), compression=self.compression,
            mode=nix.FileMode.Overwrite)

        sec = f.create_section('config', 'configuration')
        sec['sapicore_version'] = sapicore.__version__
        sec['current_utc_time'] = str(datetime.datetime.utcnow())
        sec['git_hash'] = self.git_hash
        sec['config_data'] = yaml_dumps(self.config_data or {})

        metadata_sec = sec.create_section('metadata', 'metadata')
        for key, value in (self.metadata or {}).items():
            metadata_sec[key] = yaml_dumps(value)

    def close_file(self) -> None:
        """Closes the nix file.
        """
        if self.nix_file is not None:
            self.nix_file.close()
            self.nix_file = None

    def create_block(self, name: str) -> None:
        """Creates a new block using the given that could subsequently be used
        with :meth:`get_property_arrays` to log to this block.

        It is used to organize the data into sessions.
        """
        self.nix_file.create_block(f'{name}_data_shape_len', 'shape')
        self.nix_file.create_block(f'{name}_data_shape', 'shape')
        self.nix_file.create_block(f'{name}_data_log', 'data')
        self.nix_file.create_block(f'{name}_data_counter', 'counter')

    def get_property_arrays(
            self, name: str, loggable_properties: FlatLoggableProps
    ) -> List[LogDataItem]:
        """Takes the list of properties to be logged and it creates a list of
        objects that can be used with :attr:`log` to log all these properties.

        See the class for an example.

        :param name: The name of the block to which the properties will be
            logged. The block must have been created previously with
            :meth:`create_block`.
        :param loggable_properties: The list of properties (in the format of
            :func:`get_loggable_properties`) to read and log.
        :return: A list of objects that can be passed to :meth:`log` to log
            the properties.
        """
        shape_len_block: nix.Block = self.nix_file.blocks[
            f'{name}_data_shape_len']
        shape_block = self.nix_file.blocks[f'{name}_data_shape']
        data_block = self.nix_file.blocks[f'{name}_data_log']
        counter_block = self.nix_file.blocks[f'{name}_data_counter']

        property_data = []
        for names, obj, prop, selection in loggable_properties:
            if not selection:
                continue

            value = getattr(obj, prop)
            tag = ':'.join(names)

            if isinstance(value, torch.Tensor):
                dtype = value.cpu().numpy().dtype
            else:
                if isinstance(value, (int, float)):
                    dtype = np.float64
                else:
                    raise ValueError(
                        f'Unrecognized data type for value <{value}>, with '
                        f'data type {type(value)} for {obj} property {prop}')

            if tag in data_block.data_arrays:
                data = data_block.data_arrays[tag]
                counter = counter_block.data_arrays[tag]
                shape_len = shape_len_block.data_arrays[tag]
                shape = shape_block.data_arrays[tag]
            else:
                data = data_block.create_data_array(
                    tag, 'data', dtype=dtype, data=[])
                counter = counter_block.create_data_array(
                    tag, 'counter', dtype=np.int64, data=[])
                shape_len = shape_len_block.create_data_array(
                    tag, 'shape_len', dtype=np.int8, data=[])
                shape = shape_block.create_data_array(
                    tag, 'shape', dtype=np.int32, data=[])

            property_data.append((obj, prop, counter, data, shape_len, shape))

        return property_data

    @staticmethod
    def log(property_arrays: List[LogDataItem], counter: int) -> None:
        """Logs all the properties passed to :meth:`get_property_arrays`.

        See the class for an example.

        :param property_arrays: The list of objects returned by
            :meth:`get_property_arrays`
        :param counter: An arbitrary integer that gets logged with the
            properties that could be later used to identify the data,
            e.g. the epoch when it was logged.
        """
        for obj, prop, counter_array, data, shape_len, shape in property_arrays:
            counter_array.append(counter)

            value = getattr(obj, prop)
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            value = np.asarray(value)

            n = len(value.shape)
            shape_len.append(n)
            if n:
                shape.append(value.shape)
            data.append(np.ravel(value))


class NixLogReader:
    """A reader that can read and return the data written to a nix file using
    :class:`NixLogWriter`.

    E.g.:

    . code-block:: python

        with NixLogReader('log.h5') as reader:
            print('Logged experiments: ', reader.get_experiment_names())
            # prints e.g. `Logged experiments:  ['example']`
            print('Logged properties: ',
                reader.get_experiment_property_paths('example'))
            # prints e.g. `Logged properties:  [('neuron_1', 'activation'), ...`
            print('Activation: ', reader.get_experiment_property_data(
                'example', ('neuron_1', 'activation')))
            # prints `Activation:  (array([0, 1, 2], dtype=int64), [array([0...`
    """

    nix_file: Optional[nix.File] = None
    """The internal nix file object."""

    filename: Union[str, Path] = ''
    """The filename of the nix file that will be opened when
    :meth:`open_file` is called.
    """

    def __init__(self, filename: Union[str, Path]):
        self.filename = filename

    def __enter__(self):
        self.open_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_file()

    def open_file(self):
        """Opens the file named in :attr:`filename` that was created with
        :meth:`NixLogWriter.create_file`.

        :meth:`close_file` must be called after this to close the file.

        An alternative syntax is to use it as a context manager, which is a
        little safer:

        . code-block:: python

            with NixLogReader('log.h5') as reader:
                ...

        This internally calls :meth:`open_file` and :meth:`close_file` as
        needed.
        """
        self.nix_file = nix.File.open(
            str(self.filename), mode=nix.FileMode.ReadOnly)

    def close_file(self):
        """Closes the nix file.
        """
        if self.nix_file is not None:
            self.nix_file.close()
            self.nix_file = None

    def get_experiment_names(self) -> List[str]:
        """Returns the list of block names found in the file that was created
        with :meth:`NixLogWriter.create_block`.

        E.g.:

        . code-block:: python

            >>> with NixLogReader('log.h5') as reader:
            >>>     print(reader.get_experiment_names())
            ['example']
        """
        f = self.nix_file
        return [
            block.name[:-9]
            for block in f.blocks if block.name.endswith('_data_log')]

    def get_experiment_property_paths(self, name: str) -> List[Tuple[str]]:
        """Returns the list of properties found in the file that was created
        with :meth:`NixLogWriter.get_property_arrays`.

        Each item in the list is itself a tuple of strings. This tuple
        represents the path to the property starting from the root object (e.g.
        the model) as attributes. E.g. ``model.child.prop`` will be represented
        as ``('model', 'child', 'prop')``.

        :param name: The name of the block to scan for properties.

        E.g.:

        . code-block:: python

            >>> with NixLogReader('log.h5') as reader:
            >>>     print(reader.get_experiment_property_paths('example'))
            [('neuron_1', 'activation'), ('neuron_1', 'intensity'),
            ('synapse', 'activation'), ('neuron_2', 'activation'),
            ('neuron_2', 'intensity'), ('activation_sum',)]
        """
        data_block: nix.Block = self.nix_file.blocks[f'{name}_data_log']
        properties = []
        for arr in data_block.data_arrays:
            properties.append(tuple(arr.name.split(':')))

        return properties

    def get_experiment_property_data(
            self, name: str, property_path: Tuple[str, ...]
    ) -> Tuple[np.ndarray, List[Union[np.ndarray, float, int]]]:
        """Returns all the data of a property previously logged with
        :meth:`NixLogWriter.log`.

        :param name: The name of the block that contains the data.
        :param property_path: The full attribute path to the property, so we
            can locate it in the data. It is an item from the list returned by
            :meth:`get_experiment_property_paths`.
        :return: A 2-tuple of ``(counter, data)``. ``counter`` and ``data``,
            have the same length. ``counter`` is an array containing the count
            value that corresponds the each data item as passed to
            the ``counter`` parameter in :meth:`NixLogWriter.log`.
            ``data`` is a list of the values logged for this property, one item
            for each call to :meth:`NixLogWriter.log`. It can be scalars or
            numpy arrays.

        E.g.:

        . code-block:: python

            >>> with NixLogReader('log.h5') as reader:
            >>>     print(reader.get_experiment_property_data(
            ...           'example', ('neuron_1', 'activation')))
            (array([0, 1, 2], dtype=int64), [
                array([0.56843126, 1.0845224 , 1.3985955 ], dtype=float32),
                array([0.40334684, 0.83802634, 0.7192576 ], dtype=float32),
                array([0.40334353, 0.59663534, 0.18203649], dtype=float32)
            ])
        """
        shape_len_block: nix.Block = self.nix_file.blocks[
            f'{name}_data_shape_len']
        shape_block = self.nix_file.blocks[f'{name}_data_shape']
        data_block = self.nix_file.blocks[f'{name}_data_log']
        counter_block = self.nix_file.blocks[f'{name}_data_counter']

        tag = ':'.join(property_path)

        counter = np.asarray(counter_block.data_arrays[tag])

        data = np.asarray(data_block.data_arrays[tag])
        shape_len = np.asarray(shape_len_block.data_arrays[tag])
        shape = np.asarray(shape_block.data_arrays[tag])

        items = [0, ] * len(shape_len)
        shape_s = 0
        data_s = 0
        for i, n in enumerate(shape_len):
            if not n:
                items[i] = data[data_s].item()
                data_s += 1
                continue

            dim = shape[shape_s:shape_s + n]
            k = np.multiply.reduce(dim)
            items[i] = np.reshape(data[data_s:data_s + k], dim)

            shape_s += n
            data_s += k

        return counter, items
