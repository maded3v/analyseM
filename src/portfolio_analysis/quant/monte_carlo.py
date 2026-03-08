from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    samples: pd.DataFrame
    best_sharpe: pd.Series
    min_volatility: pd.Series


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
