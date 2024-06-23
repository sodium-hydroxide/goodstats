"""Source code for blairtools.test"""

from ._utils import pd, np, time_now

class _Test:
    def __init__(self,
            data:pd.DataFrame,
            results:dict
            ) -> None:
        self.data = data
        self.results = results

    def __repr__(self) -> str:
        return "\n".join([
            "".join([key, "  :=   ", str(self.results[key])])
            for key in self.results.keys()
        ])


class _TTest(_Test):
    def __init__(self,
            data: pd.DataFrame,
            results: dict,
            var:str,
            group:str
            ) -> None:
        super().__init__(data, results)
        self.var = var
        self.group = group


def _one_sample_t_test(
        data:pd.DataFrame,
        value_var:str, 
        mean:float=0.,
        sig_level:float=0.05,
        null_hypothesis:str="two-sided"
        ) -> dict:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        value_var (str): _description_
        mean (float, optional): _description_. Defaults to 0..
        sig_level (float, optional): _description_. Defaults to 0.05.
        null_hypothesis (str, optional): _description_. Defaults to "two-sided".

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        pd.Series: _description_
    """
    if null_hypothesis not in ("two-sided", "lower", "upper"):
        raise ValueError("".join([
            "'null_hypothesis' must be one of the following:\n",
            "\"two-sided\": Mean is equal to specified value 'mean'",
            "\"lower\": Mean is less than the specified value 'mean'",
            "\"upper\": Mean is greater than the specified value 'mean'",
        ]))
    if sig_level < 0. or sig_level > 1.:
        raise ValueError("'sig_level' must be in [0,1]")
    
    result:dict = {
        "test_type"      : "One Sample T-Test",
        "null_hypothesis": null_hypothesis,
        "value_hyp"      : mean,
        "time"           : time_now(),
        "df"             : None,
        "num_obs"        : None,
        "value_obs"      : None,
        "se_obs"         : None,
        "t_statistic"    : None,
        "p_value"        : None,
        "sig_level"      : None,
        "t_critical"     : None,
        "lower_ci"       : None,
        "upper_ci"       : None
    }

    result["null_hypothesis"] = null_hypothesis
    result["value_hyp"]       = mean
    result["time"]            = time_now()
    
    test_data = data[value_var].values
    
    result["num_obs"]   = test_data.shape[0]
    result["df"]        = result["num_obs"] - 1
    result["value_obs"] = np.mean(test_data)
    result["se_obs"]    = (np.std(test_data, ddof=1)
                           / np.sqrt(result["num_obs"]))

    result["t_statistic"] = (result["value_obs"] - mean) / result["se_obs"]

    # Add rest of dict values
    return result




