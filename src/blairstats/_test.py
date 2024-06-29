"""Source code for blairstats.test"""
import pandas as pd
from ._utils import save_generic

class _Test:
    def __init__(self,
            data:pd.DataFrame,
            results:dict
            ) -> None:
        self.data = data
        self.results = results

    def __repr__(self) -> str:
        return "\n".join([
            f"{key}  :=   {self.results[key]}"
            for key in self.results.keys()
        ])

    def save(self, path:str) -> None:
        class_dict = self.__dict__
        save_generic(class_dict, path)
