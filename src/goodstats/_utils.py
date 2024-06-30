"""Utilities for blairstats"""

import pickle
from typing import Any
from numpy import ndarray, dtype, float64, int64, sqrt, floor, ceil
from datetime import datetime, UTC
from scipy import special # type: ignore

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

def _column_wrap(n:int, long:bool=False) -> int64:
    """Determine when to wrap facet grid

    Args:
        n (int): Number of facets being considered
        long (bool, optional): If true, make the facet grid wider than
        tall. Defaults to False.

    Returns:
        int64: Number of cells to wrap at.
    """
    c1 = int64(sqrt(min(
        [i for i in range(n,n**2)
         if (lambda a: floor(a) == a)(sqrt(i))]
    )))
    c2 = int64(ceil(n / c1))
    
    if long:
        return min(c1,c2)
    
    return max(c1,c2)

DISTRIBUTION = {
    "normal": lambda mu, sigma: {
        "cdf": lambda x: special.ndtr((x - mu) / sigma)
    },
    "student-t": lambda df: {
        "cdf": lambda x: special.stdtr(df, x)
    }
}
# dict[str,
#                    Callable[[Union[float, Any]],
#                             dict[str,
#                                  Callable[[ArrayFloat],ArrayFloat]]]] = 
