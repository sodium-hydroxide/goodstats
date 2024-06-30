"""_summary_
"""

# F401 errors are being ignored as the functions being imported are
# meant to be available in the package upon being loaded
from ._model_ols import ordinary_least_squares # noqa: F401
from ._model_curvefit import nonlinear_fit # noqa: F401