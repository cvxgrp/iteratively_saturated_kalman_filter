from scipy.stats import chi2


def chi_squared_quantile(quantile: float, degrees_of_freedom: int) -> float:
    """
    Calculates the chi-squared quantile (inverse CDF) for a given probability.

    Args:
        quantile: The probability (quantile) for which to find the value.
        degrees_of_freedom: The degrees of freedom of the chi-squared distribution.

    Returns:
        The chi-squared value.
    """
    return chi2.ppf(quantile, degrees_of_freedom)
