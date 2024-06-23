"""Utilities for blairstats"""

# ruff: disable=unused-import
# pylint: disable=no-name-in-module
import pickle
import typing as tp
from datetime import datetime, UTC

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import (
    ndtr,       # CDF: Normal
    ndtri,      #  QF: Normal
    stdtr,      # CDF: Student T
    stdtrit,    #  QF: Student T
    nctdtr,     # CDF: Non-Central T
    nctdtrit,   #  QF: Non-Central T
    chdtr,      # CDF: Chi-Squared
    chdtri,     #  QF: Chi-Squared
    chndtr,     # CDF: Non-Central Chi-Squared
    chndtrix,   #  QF: Non-Central Chi-Squared
    fdtr,       # CDF: F-Distribution
    fdtri       #  QF: F-Distribution
)
# pylint: enable=no-name-in-module
# pylint: enable=unused-import

def time_now() -> str:
    """Return current UTC time in ISO 8601 formatted string"""
    time = datetime.now(UTC).replace(microsecond=0).isoformat()
    return str(time)
