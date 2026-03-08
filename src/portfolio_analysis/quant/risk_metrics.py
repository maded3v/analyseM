from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskReport:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and abs(denominator) > 1e-12 else 0.0


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        raise ValueError("returns series is empty")

    compounded_growth = (1.0 + returns).prod()
    n_periods = returns.shape[0]
    if n_periods == 0:
        return 0.0
    return float(compounded_growth ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        raise ValueError("returns series is empty")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    return _safe_divide(ann_ret - risk_free_rate, ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if returns.empty:
        raise ValueError("returns series is empty")

    downside = returns[returns < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if not downside.empty else 0.0
    ann_ret = annualized_return(returns, periods_per_year)
    return _safe_divide(ann_ret - risk_free_rate, downside_dev)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        raise ValueError("returns series is empty")

    wealth_index = (1.0 + returns).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = wealth_index / running_peak - 1.0
    return float(drawdown.min())


def value_at_risk_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if returns.empty:
        raise ValueError("returns series is empty")

    alpha = 1.0 - confidence
    return float(np.percentile(returns, 100 * alpha))


def conditional_var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    var_value = value_at_risk_historical(returns, confidence)
    tail = returns[returns <= var_value]
    return float(tail.mean()) if not tail.empty else var_value


def compute_risk_report(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> RiskReport:
    """Compute key risk metrics for a return series."""
    if returns.empty:
        raise ValueError("returns series is empty")

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if clean_returns.empty:
        raise ValueError("returns series does not contain numeric values")

    return RiskReport(
        annual_return=annualized_return(clean_returns, periods_per_year),
        annual_volatility=annualized_volatility(clean_returns, periods_per_year),
        sharpe_ratio=sharpe_ratio(clean_returns, risk_free_rate, periods_per_year),
        sortino_ratio=sortino_ratio(clean_returns, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(clean_returns),
        var_95=value_at_risk_historical(clean_returns, confidence),
        cvar_95=conditional_var_historical(clean_returns, confidence),
    )
