"""Source code for blairstats.model

"""
import pandas as pd
from ._utils import save_generic

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
            endog_name:str
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

    def plot_resid_qq(self):
        pass

    def plot_resid_dist(self):
        pass

    def plot_resid_fit(self, title_name:str|None=None):
        from matplotlib.pyplot import subplots, show
        var_name:str = self.__endog_name if (title_name is None) else title_name

        fig, ax = subplots()
        ax.scatter(
            x=self.data[f"resid_{self.__endog_name}"],
            y=self.data[f"fit_{self.__endog_name}"],
            s=5,
            c="k"
        )
        ax.axhline(0,lw=1,c="k")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Fitted Values")
        ax.set_title(f"Fitted Values vs. Residuals of {var_name}")
        show()

    def predict(self, new_data: pd.DataFrame):
        pass
