from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    samples: pd.DataFrame
    best_sharpe: pd.Series
    min_volatility: pd.Series


@dataclass(frozen=True)
class PortfolioPathSimulation:
    terminal_values: pd.Series
    quantiles: dict[str, float]
    mean_terminal: float
    median_terminal: float


def simulate_random_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    n_portfolios: int = 10_000,
    risk_free_rate: float = 0.0,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo portfolio simulation with random long-only weights.

    Weights are sampled from a Dirichlet distribution to guarantee:
    1) non-negative weights
    2) sum(weights) = 1
    """
    if expected_returns.empty:
        raise ValueError("expected_returns is empty")
    if covariance.empty:
        raise ValueError("covariance matrix is empty")
    if n_portfolios <= 0:
        raise ValueError("n_portfolios must be > 0")

    assets = list(expected_returns.index)
    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    n_assets = len(assets)

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    portfolio_returns = weights @ mu
    portfolio_variances = np.einsum("ij,jk,ik->i", weights, sigma, weights)
    portfolio_volatility = np.sqrt(np.maximum(portfolio_variances, 0.0))

    sharpe = np.divide(
        portfolio_returns - risk_free_rate,
        portfolio_volatility,
        out=np.zeros_like(portfolio_returns),
        where=portfolio_volatility > 1e-12,
    )

    samples = pd.DataFrame(
        {
            "expected_return": portfolio_returns,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe,
        }
    )

    for idx, asset in enumerate(assets):
        samples[f"w_{asset}"] = weights[:, idx]

    best_sharpe_row = samples.loc[samples["sharpe_ratio"].idxmax()]
    min_volatility_row = samples.loc[samples["volatility"].idxmin()]

    return MonteCarloResult(
        samples=samples,
        best_sharpe=best_sharpe_row,
        min_volatility=min_volatility_row,
    )


def simulate_portfolio_terminal_values(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int = 252,
    n_paths: int = 3000,
    initial_value: float = 100.0,
    seed: int | None = None,
) -> PortfolioPathSimulation:
    if expected_returns.empty:
        raise ValueError("expected_returns is empty")
    if covariance.empty:
        raise ValueError("covariance matrix is empty")
    if weights.empty:
        raise ValueError("weights is empty")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    aligned_weights = weights.reindex(expected_returns.index).fillna(0.0)
    total_weight = float(aligned_weights.sum())
    if abs(total_weight) < 1e-12:
        raise ValueError("weights sum is near zero")
    aligned_weights = aligned_weights / total_weight

    daily_mean = expected_returns.to_numpy(dtype=float) / 252.0
    daily_covariance = covariance.to_numpy(dtype=float) / 252.0
    weight_vector = aligned_weights.to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    simulated_asset_returns = rng.multivariate_normal(
        mean=daily_mean,
        cov=daily_covariance,
        size=(n_paths, horizon_days),
        check_valid="warn",
    )
    portfolio_daily_returns = np.einsum("pta,a->pt", simulated_asset_returns, weight_vector)

    terminal_values = initial_value * np.prod(1.0 + portfolio_daily_returns, axis=1)
    terminal_series = pd.Series(terminal_values, name="terminal_value")

    quantiles = {
        "p05": float(np.percentile(terminal_values, 5)),
        "p25": float(np.percentile(terminal_values, 25)),
        "p50": float(np.percentile(terminal_values, 50)),
        "p75": float(np.percentile(terminal_values, 75)),
        "p95": float(np.percentile(terminal_values, 95)),
    }

    return PortfolioPathSimulation(
        terminal_values=terminal_series,
        quantiles=quantiles,
        mean_terminal=float(np.mean(terminal_values)),
        median_terminal=float(np.median(terminal_values)),
    )
