"""Source code for blairtools.test ttest functions"""

import typing as tp
import numpy as np
import pandas as pd
from .__utils__ import time_now, ArrayFloat

TTEST_HYPOTHESIS: dict[str, tp.Callable[[str,float|int], tuple[str,str]]] = {
    "two-sided": (lambda var, val:
        (f"H0:= {var} = {val}",
         f"H1:= {var} \u2260 {val}]")
    ),
    "upper": (lambda var, val:
        (f"H0:= {var} \u2264 {val}",
         f"H1:= {var} > {val}]")
    ),
    "lower": (lambda var, val:
        (f"H0:= {var} > {val}",
         f"H1:= {var} \u2265 {val}]")
    )
}
# Private Functions

def _t_test_formula_parse(formula:str) -> dict:
    """Parse t test formulas

    Raises:
        TypeError: Formula is not string
        ValueError: Formula contains both ~ and -
        ValueError: Formula contains multiple ~ or -

    Returns:
        dict: Container for variables and the method to call
    """
    if not isinstance(formula, str):
        raise TypeError("Formula must be string")
    
    def _unique_char_check(string:str, character:str) -> int:
        position_list: list[int] = [
            pos for pos,char in enumerate(string) if char == character
        ]
        if len(position_list) == 1:
            position: int = position_list[0]
        elif len(position_list) == 0:
            position: int = -1 # type: ignore
            # value is previously defined, but it is within control flow
        else:
            raise ValueError(f"String cannot contain more than one {character}")
        
        return position

    
    tilde_position: int = _unique_char_check(formula, "~")
    minus_position: int = _unique_char_check(formula, "-")
        
    if tilde_position != -1:
        if minus_position != -1:
            raise ValueError("Formula cannot contain both - and plus")
        else:
            lhs = formula[:tilde_position].replace(" ","")
            rhs = formula[tilde_position+1:].replace(" ", "")
            return {
                "var": lhs,
                "group": rhs,
                "method": "two_sample"
            }
    
    if minus_position != -1:
        lhs = formula[:minus_position].replace(" ","")
        rhs = formula[minus_position+1:].replace(" ", "")
        return {
            "var_after": lhs,
            "var_before": rhs,
            "method": "paired"
        }
    
    return {
        "var": formula,
        "method": "one_sample"
    }


def _t_test_two_sided(result:dict) -> dict:
    """Perform analysis of two-sided t-test

    Args:
        result (dict): Container for the t-statistics, degrees of
        freedom, and other relevant variables

    Returns:
        dict: Completed results summary
    """
    from scipy.special import stdtrit, stdtr # type: ignore

    result["p_value"] = 2 * (1-stdtr(result["df"], np.abs(result["t_statistic"])))
    result["t_crit_upper"] = stdtrit(result["df"], 1 - 0.5*result["sig_level"])
    result["t_crit_lower"] = -result["t_crit_upper"]
    result["upper_ci"] = result["value_obs"] + (result["se_obs"]*result["t_crit_upper"])
    result["lower_ci"] = result["value_obs"] - (result["se_obs"]*result["t_crit_upper"])

    return result

def _t_test_lower(result:dict) -> dict:
    """Perform analysis of alternative-lower t-test

    Args:
        result (dict): Container for the t-statistics, degrees of
        freedom, and other relevant variables

    Returns:
        dict: Completed results summary
    """
    from scipy.special import stdtrit, stdtr

    result["p_value"] = 1-stdtr(result["df"], result["t_statistic"])
    result["t_crit_upper"] = stdtrit(result["df"], 1 - result["sig_level"])
    result["t_crit_lower"] = -1*np.inf
    result["upper_ci"] = result["value_obs"] + (result["se_obs"]*result["t_crit_upper"])
    result["lower_ci"] = -1*np.inf

    return result

def _t_test_upper(result:dict) -> dict:
    """Perform analysis of alternative-upper t-test

    Args:
        result (dict): Container for the t-statistics, degrees of
        freedom, and other relevant variables

    Returns:
        dict: Completed results summary
    """
    from scipy.special import stdtrit, stdtr

    result["p_value"] = stdtr(result["df"], result["t_statistic"])
    result["t_crit_upper"] = np.inf
    result["t_crit_lower"] = stdtrit(result["df"], result["sig_level"])
    result["upper_ci"] = np.inf
    result["lower_ci"] = result["value_obs"] + (result["se_obs"]*result["t_crit_lower"])

    return result


def _t_test_one_sample(
        data:pd.DataFrame,
        var:str,
        mu_0:float=0.,
        sig_level:float=0.05,
        alt_hypothesis:str="two-sided"
        ) -> dict:
    """Calculate T-statistics and df for 1-sample t-test

    Args:
        data (pd.DataFrame): Dataframe to test on
        var (str): Dataframe column to find perform test on
        mu_0 (float, optional): Hypothesized mean value. Defaults to 0..
        sig_level (float, optional): Significance level to test at. 
        Defaults to 0.05.
        alt_hypothesis (str, optional): Form of alternative hypothesis.
        Defaults to "two-sided". Can also be "lower" or "upper"

    Returns:
        dict: Partially completed t-test
    """
    hypothesis = TTEST_HYPOTHESIS[alt_hypothesis](f"E[{var}]", mu_0)
    
    result:dict = {
        "test_type"      : "One Sample T-Test",
        "null_hypothesis": hypothesis[0],
        "alt_hypothesis" : hypothesis[1],
        "time"           : time_now(),
        "df"             : None,
        "num_obs"        : None,
        "value_obs"      : None,
        "se_obs"         : None,
        "t_statistic"    : None,
        "p_value"        : None,
        "sig_level"      : sig_level,
        "t_crit_lower"   : None,
        "t_crit_upper"   : None,
        "lower_ci"       : None,
        "upper_ci"       : None
    }

    test_data: ArrayFloat = np.array(data[var].values)

    result["num_obs"] = test_data.shape[0]
    result["df"] = result["num_obs"] - 1
    result["value_obs"] = np.mean(test_data)
    result["se_obs"] = (np.std(test_data, ddof=1) / np.sqrt(result["num_obs"]))
    result["t_statistic"] = (result["value_obs"] - mu_0) / result["se_obs"]

    return result

def _t_test_pooled(
        data:pd.DataFrame,
        var:str,
        group:str,
        mu_0:float=0.,
        sig_level:float=0.05,
        alt_hypothesis:str="two-sided"
        ) -> dict:
    """Calculate T-statistics and df for pooled t-test

    Args:
        data (pd.DataFrame): Dataframe to test on
        var (str): Dataframe column to find perform test on
        group (str): Dataframe column containing different groups
        mu_0 (float, optional): Hypothesized mean value. Defaults to 0..
        sig_level (float, optional): Significance level to test at. 
        Defaults to 0.05.
        alt_hypothesis (str, optional): Form of alternative hypothesis.
        Defaults to "two-sided". Can also be "lower" or "upper"

    Raises:
        ValueError: If the number of groups is not two

    Returns:
        dict: Partially completed t-test
    """
    hypothesis = TTEST_HYPOTHESIS[alt_hypothesis](
        f"E[{var}|{group}]-E[{var}|\u00AC{group}|]",
        mu_0
    )
    
    result:dict = {
        "test_type"      : "Paired T-Test",
        "null_hypothesis": hypothesis[0],
        "alt_hypothesis" : hypothesis[1],
        "time"           : time_now(),
        "df"             : None,
        "num_obs"        : None,
        "value_obs"      : None,
        "se_obs"         : None,
        "t_statistic"    : None,
        "p_value"        : None,
        "sig_level"      : sig_level,
        "t_crit_lower"   : None,
        "t_crit_upper"   : None,
        "lower_ci"       : None,
        "upper_ci"       : None
    }

    group_values = data[group].unique()

    if len(group_values) != 2:
        raise ValueError(f"The number of groups in \"{group}\" must equal two.")

    test_data1: ArrayFloat = np.array(
        data
        .query(f"{group} == {group_values[0]}")
        [var]
        .values
    )
    test_data2: ArrayFloat = np.array(
        data
        .query(f"{group} == {group_values[1]}")
        [var]
        .values
    )

    num1: int = test_data1.shape[0]
    num2: int = test_data2.shape[0]

    result["num_obs"] = (num1, num2)
    result["df"] = num1 + num2 - 2

    var_pooled: float = (
        ((num1-1) * (np.std(test_data1, ddof=1) ** 2)
         + (num2-1) * (np.std(test_data2, ddof=1) ** 2))
        / result["df"]
    )

    result["value_obs"] = np.mean(test_data1) - np.mean(test_data2)
    result["se_obs"] = np.sqrt(
        var_pooled
        * ((num1 ** -1) + (num2 ** -1))
    )
    result["t_statistic"] = (result["value_obs"] - mu_0) / result["se_obs"]

    return result

def _t_test_welch(
        data:pd.DataFrame,
        var:str,
        group:str,
        mu_0:float=0.,
        sig_level:float=0.05,
        alt_hypothesis:str="two-sided"
        ) -> dict:
    """Calculate T-statistics and df for welch t-test

    Args:
        data (pd.DataFrame): Dataframe to test on
        var (str): Dataframe column to find perform test on
        group (str): Dataframe column containing different groups
        mu_0 (float, optional): Hypothesized mean value. Defaults to 0..
        sig_level (float, optional): Significance level to test at. 
        Defaults to 0.05.
        alt_hypothesis (str, optional): Form of alternative hypothesis.
        Defaults to "two-sided". Can also be "lower" or "upper"

    Raises:
        ValueError: If the number of groups is not two

    Returns:
        dict: Partially completed t-test
    """
    hypothesis = TTEST_HYPOTHESIS[alt_hypothesis](
        f"E[{var}|{group}]-E[{var}|\u00AC{group}|]",
        mu_0
    )
    
    result:dict = {
        "test_type"      : "Pooled T-Test",
        "null_hypothesis": hypothesis[0],
        "alt_hypothesis" : hypothesis[1],
        "time"           : time_now(),
        "df"             : None,
        "num_obs"        : None,
        "value_obs"      : None,
        "se_obs"         : None,
        "t_statistic"    : None,
        "p_value"        : None,
        "sig_level"      : sig_level,
        "t_crit_lower"   : None,
        "t_crit_upper"   : None,
        "lower_ci"       : None,
        "upper_ci"       : None
    }

    group_values = data[group].unique()

    if len(group_values) != 2:
        raise ValueError(f"The number of groups in \"{group}\" must equal two.")

    test_data1: ArrayFloat = np.array(
        data
        .query(f"{group} == {group_values[0]}")
        [var]
        .values
    )
    test_data2: ArrayFloat = np.array(
        data
        .query(f"{group} == {group_values[1]}")
        [var]
        .values
    )

    num1: int = test_data1.shape[0]
    num2: int = test_data2.shape[0]

    result["num_obs"] = (num1, num2)

    se1: float = np.std(test_data1, ddof=1) / np.sqrt(num1)
    se2: float = np.std(test_data2, ddof=1) / np.sqrt(num2)

    cvalue: float = 1. / (
        1. + ((se2 / se1) ** 2)
    )
    result["df"] = (
        ((num1 - 1) * (num2 - 1))
        / (
            (num2 - 1)*(cvalue**2)
            + (num1 - 1)*(1 - (cvalue ** 2))
        )
    )

    result["value_obs"] = np.mean(test_data1) - np.mean(test_data2)
    result["se_obs"] = np.sqrt(se1**2 + se2**1)
    result["t_statistic"] = (result["value_obs"] - mu_0) / result["se_obs"]

    return result

def _t_test_paired(
        data:pd.DataFrame,
        var_before:str,
        var_after:str,
        mu_0:float=0.,
        sig_level:float=0.05,
        alt_hypothesis:str="two-sided"
        ) -> dict:
    """Calculate T-statistics and df for paired t-test

    Args:
        data (pd.DataFrame): Dataframe to test on
        var_before, var_after (str): Dataframe columns to find perform
        test on. Before and after corresponds to before and after some
        treatment. These are currently treated as separate columns.
        mu_0 (float, optional): Hypothesized mean value. Defaults to 0..
        sig_level (float, optional): Significance level to test at. 
        Defaults to 0.05.
        alt_hypothesis (str, optional): Form of alternative hypothesis.
        Defaults to "two-sided". Can also be "lower" or "upper"

    Returns:
        dict: Partially completed t-test
    """
    hypothesis = TTEST_HYPOTHESIS[alt_hypothesis](
        f"E[{var_after}]-E[{var_before}|]",
        mu_0
    )
    
    result:dict = {
        "test_type"      : "Paired T-Test",
        "null_hypothesis": hypothesis[0],
        "alt_hypothesis" : hypothesis[1],
        "time"           : time_now(),
        "df"             : None,
        "num_obs"        : None,
        "value_obs"      : None,
        "se_obs"         : None,
        "t_statistic"    : None,
        "p_value"        : None,
        "sig_level"      : sig_level,
        "t_crit_lower"   : None,
        "t_crit_upper"   : None,
        "lower_ci"       : None,
        "upper_ci"       : None
    }

    difference: ArrayFloat = (
        np.array(data[var_after].values)
        - np.array(data[var_before].values)
    )
    

    result["num_obs"] = difference.shape[0]
    result["df"] = result["num_obs"] - 1

    result["value_obs"] = np.mean(difference)
    result["se_obs"] = np.std(difference, ddof=1) / np.sqrt(result["num_obs"])
    result["t_statistic"] = (result["value_obs"] - mu_0) / result["se_obs"]

    return result

# Public Functions

def t_test(
        data:pd.DataFrame,
        formula:str,
        mu_0:float=0.,
        sig_level:float=0.05,
        alt_hypothesis:str="two-sided",
        eq_var:bool=True
        ) -> pd.Series:
    """Perform T-Test

    Args:
        data (pd.DataFrame): Dataframe to test on

        formula (str): Description of test variables. For a univariate
        t-test, this is just "variable" name. For a bivariate t-test,
        this is "variable ~ group" where group is an identifier for the
        group and variable is still the variable being tested. For paired
        t-tests, this is "variable_after - variable_before" where these
        are both numeric columns in the dataframe. This will raise an error
        if the formula contains both a '-' and a '~' or if two of these
        are given.

        mu_0 (float, optional): Hypothesized mean value or difference.
        Defaults to 0..

        sig_level (float, optional): Significance level to test at.
        Defaults to 0.05.

        alt_hypothesis (str, optional): Form of alternative hypothesis.
        Defaults to "two-sided". Can also be "lower" or "upper".

        eq_var (bool, optional): If true, two variable t-tests will
        perform a pooled t-test. If false, two variable t-tests will
        perform a Welch t-test. Does not matter for single variable or
        paired t-tests. Defaults to True.

    Raises:
        ValueError: If the alternative hypothesis is not a valid value
        ValueError: If the significance level is outside [0,1]

    Returns:
        TTest: TTest class containing the results and original dataset
    """
    if alt_hypothesis not in ("two-sided", "lower", "upper"):
        raise ValueError(
            "'alt_hypothesis' must be one of the following:\n"
            "\"two-sided\": Mean is equal to specified value 'mean'"
            "\"lower\": Mean is less than the specified value 'mean'"
            "\"upper\": Mean is greater than the specified value 'mean'"
        )
    if sig_level < 0. or sig_level > 1.:
        raise ValueError("'sig_level' must be in [0,1]")
    
    formula_parsed = _t_test_formula_parse(formula)
    
    if formula_parsed["method"] == "two_sample":
        if eq_var:
            result = _t_test_pooled(
                data,
                var=formula_parsed["var"],
                group=formula_parsed["group"],
                mu_0=mu_0,
                sig_level=sig_level,
                alt_hypothesis=alt_hypothesis
            )
        else:
            result = _t_test_welch(
                data,
                var=formula_parsed["var"],
                group=formula_parsed["group"],
                mu_0=mu_0,
                sig_level=sig_level,
                alt_hypothesis=alt_hypothesis
            )
    elif formula_parsed["method"] == "paired":
        result = _t_test_paired(
            data,
            var_before=formula_parsed["var_before"],
            var_after=formula_parsed["var_after"],
            mu_0=mu_0,
            sig_level=sig_level,
            alt_hypothesis=alt_hypothesis
        )
    else:
        result = _t_test_one_sample(
            data,
            var=formula_parsed["var"],
            mu_0=mu_0,
            sig_level=sig_level,
            alt_hypothesis=alt_hypothesis
        )
    
    if alt_hypothesis == "two-sided":
        result = _t_test_two_sided(result)
    elif alt_hypothesis == "lower":
        result = _t_test_lower(result)
    else:
        result = _t_test_upper(result)
    
    return pd.Series(result)


