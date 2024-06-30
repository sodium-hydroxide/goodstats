"""Regression models

Functions:
    nonlinear_fit
    ordinary_least_squares

"""
# F401 errors are being ignored as the functions being imported are
# meant to be available in the package upon being loaded
from .__model_ols__ import ordinary_least_squares # noqa: F401
from .__model_curvefit__ import nonlinear_fit # noqa: F401
