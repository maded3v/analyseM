from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def returns_from_prices(
    prices: pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute asset returns from a price matrix indexed by date.

    Args:
        prices: DataFrame with dates as index and tickers as columns.
        method: Return type, either simple returns or log returns.
        dropna: Whether to drop rows with NaN values after return calculation.

    Returns:
        DataFrame of asset returns.
    """
    if prices.empty:
        raise ValueError("prices DataFrame is empty")

    numeric_prices = prices.apply(pd.to_numeric, errors="coerce")
    if method == "simple":
        returns = numeric_prices.pct_change()
    elif method == "log":
        returns = np.log(numeric_prices / numeric_prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")

    if dropna:
        returns = returns.dropna(how="any")
    return returns


def compute_correlation_matrix(
    returns: pd.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
) -> pd.DataFrame:
    """Build correlation matrix for asset returns."""
    if returns.empty:
        raise ValueError("returns DataFrame is empty")
    return returns.corr(method=method)


def compute_covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build covariance matrix from returns.

    Covariance can be annualized via multiplication by periods_per_year.
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty")

    cov = returns.cov()
    if annualize:
        cov = cov * periods_per_year
    return cov
