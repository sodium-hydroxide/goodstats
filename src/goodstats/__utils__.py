"""Utilities for blairstats"""

import pickle
import typing
import numpy as np
import numpy.typing as npt
import pandas as pd
from datetime import datetime, UTC
from scipy import special # type: ignore

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

def _column_wrap(n:int, long:bool=False) -> np.int64:
    """Determine when to wrap facet grid

    Args:
        n (int): Number of facets being considered
        long (bool, optional): If true, make the facet grid wider than
        tall. Defaults to False.

    Returns:
        np.int64: Number of cells to wrap at.
    """
    c1 = np.int64(np.sqrt(min(
        [i for i in range(n,n**2)
         if (lambda a: np.floor(a) == a)(np.sqrt(i))]
    )))
    c2 = np.int64(np.ceil(n / c1))
    
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



class _Regression:
    """Generic Class For Regression Models

    Attributes:

    Methods:

    """
    def __init__(self,
            data: pd.DataFrame,
            summary: pd.Series,
            parameters: pd.DataFrame,
            model,
            endog_name:str,
            exog_names:list[str]|str
            ) -> None:
        """Constructor for generic regression class

        Args:
            data (pd.DataFrame): Dataset model was trained on

            summary (pd.Series): Summary of model performance

            parameters (pd.DataFrame): Estimators, uncertainties, and statistics for fitted parameters

            model: Model object (from statsmodels, scipy, or sklearn)
        """
        self.data = data
        self.summary = summary
        self.parameters = parameters
        self.model = model
        self.__endog_name:str = endog_name
        self.__exog_names:list[str] | str = exog_names
        pass

    def __repr__(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return (
            f"Summary:\n{self.summary.__repr__()}\n"
            f"Parameters:\n{self.parameters.__repr__()}"
        )

    def save(self, path:str) -> None:
        """_summary_

        Args:
            path (str): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
        class_dict = self.__dict__
        save_generic(class_dict, path)

    def plot(self,
            kind:typing.Literal["resid_qq",
                                "resid_dist",
                                "resid_fit",
                                "fit_obs"],
            path:str|None=None,
            title_name:str|None=None
            ) -> None:
        """_summary_

        Args:
            kind (typing.Literal[&quot;resid_qq&quot;, &quot;resid_dist&quot;, &quot;resid_fit&quot;, &quot;fit_obs&quot;]): _description_
            path (str | None, optional): _description_. Defaults to None.
            title_name (str | None, optional): _description_. Defaults to None.
        """
        {
            "resid_qq":self.__plot_resid_qq,
            "resid_dist": self.__plot_resid_dist,
            "resid_fit": self.__plot_resid_fit,
            "fit_obs": self.__plot_fit_obs
        }[kind](path, title_name)

    def __plot_resid_qq(self,
            path:str|None=None,
            title_name:str|None=None
        ) -> None:
        """_summary_

        Args:
            path (str | None, optional): _description_. Defaults to None.
            title_name (str | None, optional): _description_. Defaults to None.
        """
        from statsmodels.api import ProbPlot, qqline # type: ignore
        from matplotlib.pyplot import subplots, show, savefig
        var_name:str = self.__endog_name if (title_name is None) else title_name

        fig, ax = subplots()
        ax.set_title(f"QQ of Residuals of {var_name}")

        pp = ProbPlot(self.data[f"resid_{self.__endog_name}"], fit=True)
        qq = pp.qqplot(marker='.', markerfacecolor='k', markeredgecolor='k', alpha=1, ax=ax)
        qqline(qq.axes[0], line='45', fmt='k', lw=1)

        if path is None:
            show()
        else:
            savefig(path)

    def __plot_resid_dist(self,
            path:str|None=None,
            title_name:str|None=None
        ) -> None:
        """_summary_

        Args:
            path (str | None, optional): _description_. Defaults to None.
            title_name (str | None, optional): _description_. Defaults to None.
        """
        from matplotlib.pyplot import subplots, show, savefig
        from seaborn import histplot # type: ignore
        var_name:str = self.__endog_name if (title_name is None) else title_name

        fig, ax = subplots()
        histplot(self.data,
                 x=f"resid_std_{self.__endog_name}",
                 stat="density",
                 kde=True,
                 color="k",
                 ax=ax)
        ax.set_xlabel("Studentized Residual")
        ax.set_title(f"Distribution of Residuals of {var_name}")

        if path is None:
            show()
        else:
            savefig(path)

    def __plot_resid_fit(self,
            path:str|None=None,
            title_name:str|None=None
        ) -> None:
        """_summary_

        Args:
            path (str | None, optional): _description_. Defaults to None.
            title_name (str | None, optional): _description_. Defaults to None.
        """
        from matplotlib.pyplot import subplots, show, savefig
        var_name:str = self.__endog_name if (title_name is None) else title_name

        fig, ax = subplots()
        ax.scatter(
            x=self.data[f"fit_{self.__endog_name}"],
            y=self.data[f"resid_{self.__endog_name}"],
            s=5,
            c="k"
        )
        ax.axhline(0,lw=1,c="k")
        ax.set_ylabel("Residuals")
        ax.set_xlabel("Fitted Values")
        ax.set_title(f"Residuals vs. Fitted Values of {var_name}")

        if path is None:
            show()
        else:
            savefig(path)

    def __plot_fit_obs(self,
            path:str|None=None,
            title_name:str|None=None
        ) -> None:
        """_summary_

        Args:
            path (str | None, optional): _description_. Defaults to None.
            title_name (str | None, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        from matplotlib.pyplot import subplots, show, grid, savefig
        from numpy import abs, argsort, array
        endog_name: str = self.model.endog_names
        plot_data = pd.DataFrame(self.model.exog,
                                 columns=self.model.exog_names)
        plot_data = plot_data[[col for col in plot_data.columns if col != "Intercept"]]
        exog_names = list(plot_data.columns)
        if len(exog_names) != 1:
            raise NotImplementedError("Only possible for one independent variable")

        plot_data.insert(0, column=endog_name, value=self.model.endog)
        plot_data = pd.concat([
            (self.data[
                [f"fit_{endog_name}",
                 f"resid_{endog_name}",
                 f"resid_std_{endog_name}"]]
             .assign(**{
                "se_fit": (lambda df: abs(
                    df[f"resid_std_{endog_name}"]
                    / df[f"resid_{endog_name}"])),
                "lower1": (lambda df:
                    df[f"fit_{endog_name}"]
                    - df["se_fit"]),
                "upper1": (lambda df:
                    df[f"fit_{endog_name}"]
                    + df["se_fit"]),
                "lower2": (lambda df:
                    df[f"fit_{endog_name}"]
                    - 2*df["se_fit"]),
                "upper2": (lambda df:
                    df[f"fit_{endog_name}"]
                    + 2*df["se_fit"])
             })
             .drop(columns=[f"resid_{endog_name}",
                            f"resid_std_{endog_name}",
                            "se_fit"])
            ),
            plot_data],
            axis=1
        )

        fig, ax = subplots()
        grid(which="both",axis="both")

        x = array(plot_data[exog_names[0]].values)
        order = argsort(x)
        x=x[order]

        yobs=plot_data[endog_name].values
        yobs=yobs[order]

        yfit=plot_data[f"fit_{endog_name}"].values
        yfit=yfit[order]

        yl1=plot_data["lower1"].values
        yl1=yl1[order]
        yu1=plot_data["upper1"].values
        yu1=yu1[order]

        yl2=plot_data["lower2"].values
        yl2=yl2[order]
        yu2=plot_data["upper2"].values
        yu2=yu2[order]

        ax.scatter(x, yobs,
                   facecolors="none",
                   edgecolors="k",
                   s=40,
                   label="Observed")
        ax.plot(x, yfit,
                color="k",
                lw=1,
                label="Fitted")
        ax.fill_between(x, yl1, yu1,
                        color="k",
                        alpha=0.2,
                        label="1\u03C3")
        ax.fill_between(x, yl2, yu1,
                        color="k",
                        alpha=0.1,
                        label="2\u03C3")
        ax.set_xlabel(exog_names[0])
        ax.set_ylabel(endog_name)
        ax.legend()
        ax.set_title("Comparison of Fitted and Observed Values")

        if path is None:
            show()

        else:
            savefig(path)


def __parameter_summary__(
        estimator:npt.NDArray[np.float64],
        covariance_matrix:npt.NDArray[np.float64],
        names:list[str]
        ) -> pd.DataFrame:
    """Generate parameter summaries for regression models

    Args:
        estimator (npt.NDArray[np.float64]): Estimator of model parameters
        covariance_matrix (npt.NDArray[np.float64]): Covariance matrix of model
        names (list[str]): Names of parameters

    Returns:
        pd.DataFrame: DataFrame containing the estimators, standard
        errors, relative errors, correlation matrices, and t-values
    """
    # if not isinstance(estimator, npt.NDArray[np.float64]):
    #     raise TypeError("estimator must be numpy array of float")
    # if not isinstance(covariance_matrix, npt.NDArray[np.float64]):
    #     raise TypeError("covariance_matrix must be numpy array of float")
    # if not isinstance(names, list[str]):
    #     raise TypeError("names must be list of strings")

    se = np.sqrt(np.diag(covariance_matrix))

    cor = covariance_matrix / np.outer(se,se)

    params = pd.DataFrame(
        cor,
        index=names,
        columns=[f"cor_{name}" for name in names]
    )

    params = pd.concat(
        [(pd.DataFrame({"estimate":estimator, "se": se},
                      index = names)
         .assign(**{"re": (lambda df:
             df["se"] / np.abs(df["estimate"])
        )})),
        params],
        axis=1
    ).assign(**{"tvalues": (lambda df:
        df["estimate"] / df["se"]
    )})

    return params


def __error_prop_estimate__(
    func:typing.Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    exog:npt.NDArray[np.float64],
    params:npt.NDArray[np.float64],
    se_params:npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
    estimated = func(exog, params)
    
    upper = func(exog, params + se_params)
    lower = func(exog, params - se_params)
    
    u_se_estimate = np.abs(estimated - upper)
    l_se_estimate = np.abs(estimated - lower)
    
    se_estimates = (np.insert(u_se_estimate, 0, l_se_estimate)
                    .reshape(2, exog.shape[0])
                    .T)
    
    se_estimate:npt.NDArray[np.float64] = np.max(se_estimates, axis=1)
    
    return se_estimate




