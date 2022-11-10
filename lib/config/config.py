#!/usr/bin/env python3

import os
import toml


def getAbsPath(path, rootPath):
    """
    Converts the given path to an absolute path if it is not one already.
    The absolute path is obtained by combining with the given root path. 

    Parameters
    ----------
    path : str
        Path to convert to abs path if not already.
    rootPath : str
        Path to assume paths are relative to.

    Returns
    -------
    str
        Absolute path
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(rootPath, path))


class AttrDict(dict):
    """
    AttrDict allows a python dictionary to be converted into an object having
    the keys as properties returning the corresponding dictionary value. That
    is, the keys can be accessed as properties of the object using dot notation.
    The usual dictionary access methods are also supported.
    The dictionary may contain nested dictionaries or lists.
    Example:
    ```python
    d = AttrDict() # empty
    d = AttrDict({'key':'value'})
    # setting
    d.val2 = 'second'
    d['val2'] = 'second'
    # getting
    val2 = d.val2
    val2 = d['val2']
    ```
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            self[key] = self._convert(value)

    def _convert(self, obj):
        if isinstance(obj, dict):
            value = AttrDict(obj)
        elif isinstance(obj, list):
            for idx in range(len(obj)):
                obj[idx] = self._convert(obj[idx])
            value = obj
        else:
            value = obj
        return value


def loadConfig(cfgPath):
    """
    Loads toml file containing configurations for model training

    Parameters
    ----------
    cfgPath : string
        Path to config file in TOML format.

    Returns
    -------
    AttrDict
        AttrDict object created from the parsed
        config file.
    """
    with open(cfgPath, 'r') as df:
        tomlString = df.read()
    cfg = AttrDict(toml.loads(tomlString))

    return cfg
