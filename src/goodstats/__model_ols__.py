"""Source code for OLS models"""
import numpy as np
import pandas as pd
from patsy import dmatrices # type: ignore
from statsmodels.api import OLS # type: ignore
from .__model__ import _Regression, __parameter_summary__
from .__utils__ import time_now

class OrdinaryLeastSquares(_Regression):
    """_summary_

    Args:
        _Regression (_type_): _description_
    """
    def __init__(self,
            data: pd.DataFrame,
            summary: pd.Series,
            parameters: pd.DataFrame,
            model: OLS
            ) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            summary (pd.Series): _description_
            parameters (pd.DataFrame): _description_
            model (_type_): _description_
        """
        super().__init__(data, summary, parameters, model,
                         model.endog_names, model.exog_names)
        self.tests: dict[str, pd.Series]={}
    
    def ttest(self,
            variable:str,
            hypothesized:float,
            alt_hypothesis
            ) -> pd.Series:
        raise NotImplementedError
    
    def ftest(self,
            hypthesized:float,
            alt_hypothesis
            ) -> pd.Series:
        raise NotImplementedError

def _ols_params(model) -> pd.DataFrame:
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    model_fit = model.fit()
    estimates = (model_fit.params)
    cov = model_fit.cov_params()
    names = model.exog_names
    return __parameter_summary__(estimates, cov, names)

def _ols_summary(model, formula) -> pd.Series:
    """_summary_

    Args:
        model (_type_): _description_
        formula (_type_): _description_

    Returns:
        pd.Series: _description_
    """
    model_fit = model.fit()
    nobs = np.int64(model.nobs)
    df_model = np.int64(model.df_model)
    df_resid = np.int64(model.df_resid)
    df_total = df_model + df_resid

    ss_model = model_fit.mse_model * df_model
    ss_resid = model_fit.mse_resid * df_resid
    ss_total = ss_model + ss_resid

    summary = pd.Series({
        "model": f"OLS: {formula}", # type: ignore
        "time": time_now(), # type: ignore
        "nobs": nobs, # type: ignore
        "ddof_model": nobs - df_model, # type: ignore
        "ddof_resid": nobs - df_resid, # type: ignore
        "ddof_total": nobs - df_total, # type: ignore
        "rsquared": model_fit.rsquared,
        "rsquared_adj": model_fit.rsquared_adj,
        "fvalue": model_fit.fvalue,
        "ss_model": ss_model,
        "ss_resid": ss_resid,
        "ss_total": ss_total,
        "cov_type": model_fit.cov_type,
        "log_likelihood": model_fit.llf,
        "aic": model_fit.aic,
        "bic": model_fit.bic,
        # "omnibus": None, # type: ignore
        # "skew": None, # type: ignore
        # "durbin_watson": None, # type: ignore
        # "jarque-bera": None, # type: ignore
        # "cond_num": None # type: ignore
    })

    return summary

def _ols_data(model, data:pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        model (_type_): _description_
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    #raise NotImplementedError("Do this dummy")
    from statsmodels.stats.outliers_influence import OLSInfluence # type: ignore
    model_fit = model.fit()
    model_inf = OLSInfluence(model.fit())
    endog_name = model.endog_names
    resid = model_fit.resid
    fit = model_fit.fittedvalues
    original = resid + fit
    data = data.assign(**{
        f"_{endog_name}": original,
        f"fit_{endog_name}": fit,
        f"resid_{endog_name}": resid,
        f"resid_std_{endog_name}": model_inf.resid_studentized
    })
    return data

def __ols_model__(data:pd.DataFrame,
                  formula:str,
                  ) -> OLS:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        formula (str): _description_

    Returns:
        OLS: _description_
    """
    endog, exog = dmatrices(formula, data)
    model = OLS(endog, exog)
    return model

def ordinary_least_squares(
        data:pd.DataFrame,
        formula:str
        ) -> OrdinaryLeastSquares:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        formula (str): _description_

    Returns:
        OrdinaryLeastSquares: _description_
    """
    

    model = __ols_model__(data,formula)
    params = _ols_params(model)
    summary = _ols_summary(model, formula)
    data_new = _ols_data(model, data)

    ols_class = OrdinaryLeastSquares(
        data_new,
        summary,
        params,
        model
    )

    return ols_class
