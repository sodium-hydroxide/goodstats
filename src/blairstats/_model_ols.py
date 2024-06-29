"""Source code for OLS models"""
import numpy as np
import pandas as pd
from ._model import _Regression
from ._utils import time_now

class OrdinaryLeastSquares(_Regression):
    def __init__(self,
            data: pd.DataFrame,
            summary: pd.Series,
            parameters: pd.DataFrame,
            model
            ) -> None:
        super().__init__(data, summary, parameters, model, model.endog_names)

def _ols_params(model) -> pd.DataFrame:
    model_fit = model.fit()
    params_vals = (model_fit.params)
    cov = model_fit.cov_params()
    names = model.exog_names
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
            "tvalues": model_fit.tvalues
        })
    )
    return params

def _ols_summary(model, formula) -> pd.Series:
    model_fit = model.fit()
    df_model = np.int64(model.df_model)
    df_resid = np.int64(model.df_resid)
    df_total = df_model + df_resid

    ss_model = model_fit.mse_model * df_model
    ss_resid = model_fit.mse_resid * df_resid
    ss_total = ss_model + ss_resid

    summary = pd.Series({
        "model": f"OLS: {formula}", # type: ignore
        "time": time_now(), # type: ignore
        "rsquared": model_fit.rsquared,
        "rsquared_adj": model_fit.rsquared_adj,
        "nobs": np.int64(model.nobs), # type: ignore
        "df_model": df_model, # type: ignore
        "df_resid": df_resid, # type: ignore
        "df_total": df_total, # type: ignore
        "ss_model": ss_model,
        "ss_resid": ss_resid,
        "ss_total": ss_total,
        "cov_type": model_fit.cov_type,
        "fvalue": model_fit.fvalue,
        "log_likelihood": model_fit.llf,
        "aic": model_fit.aic,
        "bic": model_fit.bic,
        "omnibus": None, # type: ignore
        "skew": None, # type: ignore
        "durbin_watson": None, # type: ignore
        "jarque-bera": None, # type: ignore
        "cond_num": None # type: ignore
    })

    return summary

def _ols_data(model, data:pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import OLSInfluence # type: ignore
    model_fit = model.fit()
    model_inf = OLSInfluence(model.fit())
    endog_name = model.endog_names
    resid = model_fit.fittedvalues
    fit = model_fit.resid
    original = resid + fit
    data = data.assign(**{
        f"_{endog_name}": original,
        f"fit_{endog_name}": fit,
        f"resid_{endog_name}": resid,
        f"resid_std_{endog_name}": model_inf.resid_studentized
    })
    return data

def ordinary_least_squares(data, formula):
    from patsy import dmatrices # type: ignore
    from statsmodels.api import OLS # type: ignore

    endog, exog = dmatrices(formula, data)
    model = OLS(endog, exog)
    params = _ols_params(model)
    summary = _ols_summary(model, formula)
    data_new = _ols_data(model, data)

    ols_class = (
        data_new,
        summary,
        params,
        model
    )

    return ols_class