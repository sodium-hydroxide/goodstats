"""Source code for blairstats.model

"""

from ._utils import *

class _Regression:
    def __init__(self) -> None:
        pass

    def plot_resid_qq(self):
        pass

    def plot_resid_dist(self):
        pass

    def plot_resid_fit(self):
        pass

    def predict(self, new_data):
        pass
    

class _OLS(_Regression):
    def __init__(self) -> None:
        super().__init__()

class _OLSFormula(_OLS):
    def __init__(self) -> None:
        super().__init__()

class _OLSXY(_OLS):
    def __init__(self) -> None:
        super().__init__()

def ols():
    pass