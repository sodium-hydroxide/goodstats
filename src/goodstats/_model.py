"""Source code for blairstats.model

"""
import pandas as pd
from ._utils import save_generic
import typing

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
        self.parameters = parameters,
        self.model = model
        self.__endog_name:str = endog_name
        self.__exog_names:list[str] | str = exog_names
        pass

    def __repr__(self) -> str:
        return (
            f"Summary:\n{self.summary.__repr__()}\n"
            f"Parameters:\n{self.parameters.__repr__()}"
        )
    
    def save(self, path:str) -> None:
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
        from matplotlib.pyplot import subplots, show, grid, savefig
        from numpy import abs
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
        ax.scatter(plot_data[exog_names[0]],
                   (plot_data[endog_name]),
                   facecolors="none",
                   edgecolors="k",
                   s=40,
                   label="Observed")
        ax.plot(plot_data[exog_names[0]],
                (plot_data[f"fit_{endog_name}"]),
                color="k",
                lw=1,
                label="Fitted")
        ax.fill_between(plot_data[exog_names[0]],
                        (plot_data["lower1"]),
                        (plot_data["upper1"]),
                        color="k",
                        alpha=0.2,
                        label="1\u03C3")
        ax.fill_between(plot_data[exog_names[0]],
                        (plot_data["lower2"]),
                        (plot_data["upper2"]),
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
