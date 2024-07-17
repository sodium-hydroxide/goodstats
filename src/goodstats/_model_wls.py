
import numpy as np
import pandas as pd
from patsy import dmatrices # type: ignore
from scipy.linalg import pinv # type: ignore
from ._utils import _Regression, time_now

class LeastSquares(_Regression):
    """LeastSquares Regression Models

    _extended_summary_

    Parameters
    ----------
    _Regression : Super class
        Super class for generic regression models

    Attributes
    ----------


    Methods
    -------

    """
    def __init__(
            self,
            data: pd.DataFrame,
            summary: pd.Series,
            parameters: pd.DataFrame
            ) -> None:
        """Constructor for Least Squares class

        Parameters
        ----------
        data : pd.DataFrame
            Data the model was trained on
        summary : pd.Series
            Summary statistics for the model
        parameters : pd.DataFrame
            Summary statistics for the model parameters
        """
        endog_name: str = summary["endog"]
        exog_names: str | list[str] = summary["exog"]
        super().__init__(
            data,
            summary,
            parameters,
            endog_name,
            exog_names
        )


def least_squares(
        data:pd.DataFrame,
        formula:str,
        se_actual_name: str|None = None
        ) -> LeastSquares:
    """Compute ordinary or weighted least squares fit of data

    _extended_summary_

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing the values to be fitted
    formula : str
        Formula describing the endogenous variable and exogenous variables
    se_actual_name : str | None, optional
        If none, perform ordinary least squares. Weighted least squares
        is performed when the parameter is set to the name of a dataframe
        column containing the standard errors of the endogenous variables,
        by default None

    Returns
    -------
    _type_
        Least squares fit model object for the data.
    """
    #region Initialize Model
    nobs: np.int64 = np.int64(data.shape[0])
    endog, exog = dmatrices(formula, data)
    endog_name = endog.design_info.column_names[0]
    exog_names = exog.design_info.column_names

    se_vector = (
        np.ones(nobs)
        if se_actual_name is None
        else np.array(data[se_actual_name], dtype=np.float64)
    )
    weight =np.diag(1 / np.square(se_vector))
    weight /= np.sum(weight)
    #endregion

    #region Solve Parametersprint(exog)
    inv_exog_sqr = pinv(exog.T @ weight @ exog)

    parameters = inv_exog_sqr @ exog.T @ weight
    influence = exog @ parameters
    parameters = parameters @ endog
    #endregion

    #region Get MSE and Model Statistics
    ddof_total = np.int64(1)
    ddof_resid = np.prod(parameters.shape)
    ddof_model = ddof_resid - ddof_total

    normalization = (
        weight
        .diagonal()
        .repeat(nobs)
        .reshape((nobs, nobs))
        .T
    )
    ident = np.identity(int(nobs))

    ss_total = (ident - normalization) @ endog
    ss_total = (ss_total.T @ weight @ ss_total)[0,0]

    fit = influence @ endog
    resid = endog - fit
    ss_resid = (resid.T @ weight @ resid)[0,0]

    ss_model = (influence - normalization) @ endog
    ss_model = (ss_model.T @ weight @ ss_model)[0,0]

    ms_total = ss_total / (nobs - ddof_total)
    ms_resid = ss_resid / (nobs - ddof_resid)
    ms_model = ss_model / (nobs - ddof_model)

    rsquared = 1. - (ss_resid / ss_total)
    rsquared_adj = 1. - (ms_resid / ms_total)
    fvalue = ms_model / ms_resid


    #endregion

    #region Calculation of Covariance
    cov = ms_resid * inv_exog_sqr
    se = np.sqrt(np.diag(cov))
    cor = cov / np.outer(se,se)

    #endregion

    #region Saving Data, Summary, and Parameters
    data = pd.concat([
        pd.DataFrame(endog, columns=[endog_name]),
        pd.DataFrame(
            se_vector * np.float64(se_actual_name is not None),
            columns=[f"se_{endog_name}"]
        ),
        pd.DataFrame(exog, columns=exog_names)],
        axis=1
        )
    data = data.assign(**{
        "weight": np.diag(weight),
        f"fit_{endog_name}": fit,
        f"se_fit_{endog_name}": np.sqrt(ms_resid) * se_vector,
        f"resid_{endog_name}": resid,
        f"se_resid_{endog_name}": lambda df: np.sqrt(
            df[f"se_fit_{endog_name}"]**2
            + (df[f"se_{endog_name}"]**2
               * np.float64(se_actual_name is None))
        ),
        f"std_resid_{endog_name}": lambda df: (
            df[f"resid_{endog_name}"]
            / df[f"se_resid_{endog_name}"]
        )
    })

    summary = pd.Series({
        "model": f"{"OLS" if se_actual_name is None else "WLS"}: {formula}", # type: ignore
        "endog": endog_name,
        "exog": exog_names,
        "time": time_now(), # type: ignore
        "nobs": nobs, # type: ignore
        "ddof_model": ddof_model,
        "ddof_resid": ddof_resid,
        "ddof_total": ddof_total, # type: ignore
        "rsquared": rsquared,
        "rsquared_adj": rsquared_adj,
        "fvalue": fvalue,
        "ss_model": ss_model,
        "ss_resid": ss_resid,
        "ss_total": ss_total,
        "var_resid": ms_resid
    })

    parameters = (
        pd.DataFrame(parameters, columns=["parameter"], index=exog_names)
        .assign(**{
            "se": se,
            "re": lambda df: np.abs(df["se"] / df["parameter"]),
        })
        .pipe(lambda df: pd.concat([df, pd.DataFrame(
            cor, index=exog_names,
            columns=[f"cor_{name}" for name in exog_names]
        )], axis=1))
        .assign(**{"tvalues": lambda df: 1 / df["re"]})
        .T
    )

    #endregion
    return LeastSquares(data, summary, parameters)

