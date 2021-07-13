"""Logging
==========
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

    _loggable_props_: Tuple[str] = ()

    _loggable_props: List[str] = None

    _loggable_children_: Dict[str, str] = {}

    _loggable_children: Dict[str, str] = None

    @property
    def loggable_props(self) -> List[str]:
        """A property containing the list of the names of all the properties of
        the instance that could be loggable (i.e. listed in
        ``_loggable_props_``).

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
        listed in ``_loggable_children_``).

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
        not logged).

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
        >>> read_loggable_from_object(model, False)
        {'the box': {'volume': False}, 'name': False'}
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
    be logged (i.e. those that are True) and flattens them into a single list.

    This list could be used to efficiently log all the properties that need
    to be logged, by iterating the list.

    :param obj: The object from which to get the loggables.
    :param loggable: The the loggable data read with
        :func:`read_loggable_from_object` or :func:`read_loggable_from_file`.
    :param property_path: A list of strings, indicating the object children
        names, starting from some root object that lead to the ``obj``. Should
        just be None (default) in user code.
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
        not logged).
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
        >>> d = read_loggable_from_object(model)
        >>> dump_loggable('loggable.yaml', d)
        >>> read_loggable_from_file('loggable.yaml')
        {'name': 'chair'}
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
        >>> d = read_loggable_from_object(model)
        >>> dump_config('loggable.yaml', d)
        >>> with open('loggable.yaml') as fh:
        ...     print(fh.read())

    Which prints ``name: chair``.
    """
    dump_config(filename, data)


def log_tensor_board(
        writer, loggable_properties, global_step=None, walltime=None,
        prefix=''):
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
    previously there or properties that changed.

    :param obj: The configurable object.
    :param filename: The yaml filename.
    :param default_value: A bool (True/False), indicating whether to default
        the new loggable properties to True (they are logged) or False (they are
        not logged).
    :returns: The loggable list, like returned by
        :func:`get_loggable_properties`.

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

    nix_file: Optional[nix.File] = None

    filename: Union[str, Path] = ''

    git_hash = ''

    compression = nix.Compression.Auto

    metadata: Optional[dict] = None

    config_data: Optional[dict] = None

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

    def create_file(self):
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

    def close_file(self):
        if self.nix_file is not None:
            self.nix_file.close()
            self.nix_file = None

    def create_block(self, name: str) -> None:
        self.nix_file.create_block(f'{name}_data_shape_len', 'shape')
        self.nix_file.create_block(f'{name}_data_shape', 'shape')
        self.nix_file.create_block(f'{name}_data_log', 'data')
        self.nix_file.create_block(f'{name}_data_counter', 'counter')

    def get_property_arrays(
            self, name: str, loggable_properties: FlatLoggableProps
    ) -> List[LogDataItem]:
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
                        f'Unrecognized data type for value <{value}>, with data '
                        f'type {type(value)} for {obj} property {prop}')

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

    nix_file: Optional[nix.File] = None

    filename: Union[str, Path] = ''

    def __init__(self, filename: Union[str, Path]):
        self.filename = filename

    def __enter__(self):
        self.open_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_file()

    def open_file(self):
        self.nix_file = nix.File.open(
            str(self.filename), mode=nix.FileMode.ReadOnly)

    def close_file(self):
        if self.nix_file is not None:
            self.nix_file.close()
            self.nix_file = None

    def get_experiment_names(self) -> List[str]:
        f = self.nix_file
        return [
            block.name[:-9]
            for block in f.blocks if block.name.endswith('_data_log')]

    def get_experiment_property_paths(self, name: str) -> List[Tuple[str]]:
        data_block: nix.Block = self.nix_file.blocks[f'{name}_data_log']
        properties = []
        for arr in data_block.data_arrays:
            properties.append(tuple(arr.name.split(':')))

        return properties

    def get_experiment_property_data(
            self, name: str, property_path: Tuple[str, ...]
    ) -> Tuple[np.ndarray, List[Union[np.ndarray, float, int]]]:
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
