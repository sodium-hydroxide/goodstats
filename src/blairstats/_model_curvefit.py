"""Source code for OLS models"""

from typing import Callable, Any
import numpy as np
import pandas as pd
from src.blairstats._utils import ArrayFloat, time_now
from src.blairstats._model import _Regression

class NonlinearFit(_Regression):
    def __init__(self,
            data: pd.DataFrame,
            summary: pd.Series,
            parameters: pd.DataFrame,
            model
            ) -> None:
        super().__init__(data, summary, parameters, model,
                         "y", "x")
        self.tests: dict[str, pd.Series]={}

def _nonlinear_fit_params(model) -> pd.DataFrame:
    params_vals = model["params"]
    cov = model["cov"]
    names = [f"b_{i}" for i in range(len(params_vals))]
    se = np.sqrt(np.diag(cov))
    cor = cov / np.outer(se,se)
    cor = pd.DataFrame(
        cor,
        index=names,
        columns=[f"cor_{name}" for name in names]
    )
    params = (
        pd.concat([
            (pd.DataFrame({"estimate": params_vals,"se": se},
                          index=names)
             .assign(**{"re": lambda df:df["se"] / np.abs(df["estimate"])})
            ),
            cor],
            axis=1)
        .assign(**{
            "tvalues": lambda df: df["estimate"] / df["se"]
        })
    )
    return params

def _nonlinear_summary(model, data, params) -> pd.Series:
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
        "model": (f"Nonlinear-Fit: {endog}"
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

def _nonlinear_data(model, data:pd.DataFrame, params) -> pd.DataFrame:
    endog_name = model["endog"].columns[0]
    exog_name = model["exog"].columns[0]
    xvals = data[exog_name]
    yactual = data[endog_name]
    yfit = model['curve_solution'](xvals)
    resid = yactual - yfit

    # This calculation looks at the difference of the true residuals and
    # the residuals if the model parameters were +/- one standard error
    # from their estimate. It then calculates the standard error of the
    # residuals to be the maximum difference between the true residual
    # and the residuals at plus or minus a standard error. This mathod
    # does not currently consider covariance of the parameters.
    se_resid = np.max(
        np.insert((
            np.abs(resid - (
                yactual 
                - model["curve"](xvals,
                                 params["estimate"] - params["se"])
            ))
        ),0,
        (
            np.abs(resid - (
                yactual 
                - model["curve"](xvals,
                                 params["estimate"] + params["se"])
            ))
        )).reshape(2,resid.shape[0]).T,
        axis=1
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

def _nonlinear_fit_model(
        data:pd.DataFrame,
        formula:str,
        curve:Callable[[Any],ArrayFloat],
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
        curve:Callable[[Any], ArrayFloat],
        guess:tuple[float]
        ) -> tuple:

    model:dict = _nonlinear_fit_model(
        data,
        formula,
        curve,
        guess
    )
    params = _nonlinear_fit_params(model)
    data_new = _nonlinear_data(model, data, params)
    summary = _nonlinear_summary(model, data, params)

    nonlinear_fit_class = (
        data_new,
        summary,
        params,
        model
    )

    return nonlinear_fit_class