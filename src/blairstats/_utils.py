"""Utilities for blairstats"""

import pickle
from typing import Any
from numpy import ndarray, dtype, float64, int64
from datetime import datetime, UTC

ArrayFloat = ndarray[Any, dtype[float64]]
ArrayInt = ndarray[Any, dtype[int64]]
numeric = float | int
ArrayNumeric = ArrayInt | ArrayFloat

def time_now() -> str:
    """Return current UTC time in ISO 8601 formatted string"""
    time = datetime.now(UTC).replace(microsecond=0).isoformat()
    return str(time)

def save_generic(obj:dict, path:str, method=pickle, binary=True) -> None:
    """Save dictionary

    Args:
        obj (dict): Dictionary corresponding to class
        path (str): Path to file
        method (module, optional): Format to save as. Defaults to pickle.
    """
    mode = "wb" if binary else "w"
    with open(path, mode=mode) as handle:
        method.dump(obj, handle)

def load_generic(path:str, method=pickle, binary=True) -> dict:
    """Load Dictionary

    Args:
        path (str): Path to file
        method (module, optional): Format to load from. Defaults to pickle.

    Returns:
        dict: Dictionary that was previously saved
    """
    mode = "rb" if binary else "r"
    with open(path, mode=mode) as handle:
        obj: dict = method.load(handle)

    return obj
