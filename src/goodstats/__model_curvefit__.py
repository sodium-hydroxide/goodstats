"""Source code for OLS models"""

from typing import Callable, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from .__utils__ import (
    time_now,
    __Regression__,
    __parameter_summary__,
    __error_prop_estimate__
)

class NonlinearFit(__Regression__):
    def __init__(self,
            data: pd.DataFrame,
            summary: pd.Series,
            parameters: pd.DataFrame,
            model
            ) -> None:
        super().__init__(data, summary, parameters, model,
                         model["endog"].columns[0],
                         model["exog"].columns[0])
        self.tests: dict[str, pd.Series]={}

def __nonlinear_fit_params__(model) -> pd.DataFrame:
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    estimates = model["params"]
    cov = model["cov"]
    names = [f"b_{i}" for i in range(len(estimates))]
    
    return __parameter_summary__(estimates, cov, names)

def __nonlinear_summary__(model, data, params) -> pd.Series:
    endog = model["endog"].columns[0]

    nobs = np.int64(data.shape[0])
    ddof_resid = np.int64(params.shape[0])
    ddof_total = np.int64(1)
    ddof_model = ddof_resid - ddof_total

    ss_resid = np.sum(np.square(data[f"resid_{endog}"]))
    ss_total = np.sum(np.square(data[endog] - np.mean(data[endog])))
    ss_model = ss_total - ss_resid

    ms_resid = ss_resid / (nobs - ddof_resid)
    ms_total = ss_total / (nobs - ddof_total)
    ms_model = ss_model / (nobs - ddof_model)

    rsquared = 1. - (ss_resid / ss_total)
    rsquared_adj = 1 - (ms_resid / ms_total)
    fvalue = ms_model / ms_resid

    summary = pd.Series({
        "model": (f"Nonlinear-Fit: {endog}" # type: ignore
                  f" ~ F({model["exog"].columns[0]},"
                  f"{", ".join(params.index)})"
                 ),
        "time": time_now(), # type: ignore
        "nobs": nobs, # type: ignore
        "ddof_model": ddof_model, # type: ignore
        "ddof_resid": ddof_resid, # type: ignore
        "ddof_total": ddof_total, # type: ignore
        "rsquared": rsquared,
        "rsquared_adj": rsquared_adj,
        "fvalue": fvalue,
        "ss_model": ss_model,
        "ss_resid": ss_resid,
        "ss_total": ss_total,
        # "cov_type": None,
        # "log_likelihood": None,
        # "aic": None,
        # "bic": None,
        # "omnibus": None, # type: ignore
        # "skew": None, # type: ignore
        # "durbin_watson": None, # type: ignore
        # "jarque-bera": None, # type: ignore
        # "cond_num": None # type: ignore
    })

    return summary

def __nonlinear_data__(model, data:pd.DataFrame, params) -> pd.DataFrame:
    endog_name:str = model["endog"].columns[0]
    exog_name:str = model["exog"].columns[0]
    xvals = data[exog_name].values
    yactual = data[endog_name].values
    yfit:npt.NDArray[np.float64] = model['curve_solution'](xvals)
    resid = yactual - yfit

    # This calculation looks at the difference of the true residuals and
    # the residuals if the model parameters were +/- one standard error
    # from their estimate. It then calculates the standard error of the
    # residuals to be the maximum difference between the true residual
    # and the residuals at plus or minus a standard error. This mathod
    # does not currently consider covariance of the parameters.
    se_resid = __error_prop_estimate__(
        (lambda x, y: resid - yactual + model["curve"](x, y)),
        xvals,
        params["estimate"].values,
        params["se"].values
    )
    resid_std = resid / se_resid

    data = data.assign(**{
        f"_{endog_name}": yactual,
        f"fit_{endog_name}": yfit,
        f"resid_{endog_name}": resid,
        f"se_resid_{endog_name}": se_resid,
        f"resid_std_{endog_name}": resid_std
    })

    return data

def __nonlinear_fit_model__(
        data:pd.DataFrame,
        formula:str,
        curve:Callable[[Any],npt.NDArray[np.float64]],
        guess:tuple[float]
        ) -> dict:
    from patsy import dmatrices # type: ignore
    from scipy.optimize import curve_fit # type: ignore

    design = dmatrices(f"{formula} -1", data, return_type="dataframe")
    model = {
        "curve": lambda x, params: curve(x, *params),
        "param_guess": guess,
        "endog": design[0],
        "exog":  design[1]
    }
    model["params"], model["cov"] = curve_fit(
        curve,
        model["exog"].iloc[:,0].values,
        model["endog"].iloc[:,0].values,
        p0=guess
    )
    model["curve_solution"] = lambda x: model["curve"](x, model["params"])
    
    return model

def nonlinear_fit(
        data:pd.DataFrame,
        formula:str,
        curve:Callable[[Any], npt.NDArray[np.float64]],
        guess:tuple[float]
        ) -> NonlinearFit:

    model:dict = __nonlinear_fit_model__(
        data,
        formula,
        curve,
        guess
    )
    params = __nonlinear_fit_params__(model)
    data_new = __nonlinear_data__(model, data, params)
    summary = __nonlinear_summary__(model, data, params)

    nonlinear_fit_class = NonlinearFit(
        data_new,
        summary,
        params,
        model
    )

    return nonlinear_fit_class