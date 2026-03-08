from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizationConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_free_rate: float = 0.0


@dataclass(frozen=True)
class PortfolioSolution:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass(frozen=True)
class EfficientFrontierPoint:
    target_return: float
    volatility: float
    weights: pd.Series


def _validate_inputs(expected_returns: pd.Series, covariance: pd.DataFrame) -> None:
    if expected_returns.empty:
        raise ValueError("expected_returns is empty")
    if covariance.empty:
        raise ValueError("covariance matrix is empty")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix must be square")
    if covariance.shape[0] != expected_returns.shape[0]:
        raise ValueError("expected_returns length must match covariance dimensions")


def _portfolio_stats(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    portfolio_return = float(weights @ expected_returns)
    portfolio_volatility = float(np.sqrt(weights @ covariance @ weights))
    sharpe = 0.0
    if portfolio_volatility > 1e-12:
        sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, float(sharpe)


def _build_bounds(n_assets: int, constraints: OptimizationConstraints) -> tuple[tuple[float, float], ...]:
    if constraints.min_weight > constraints.max_weight:
        raise ValueError("min_weight cannot exceed max_weight")
    return tuple((constraints.min_weight, constraints.max_weight) for _ in range(n_assets))


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if abs(total) < 1e-12:
        raise ValueError("cannot normalize weights with near-zero sum")
    return weights / total


def _run_optimization(
    objective,
    n_assets: int,
    constraints: Iterable[dict],
    bounds: tuple[tuple[float, float], ...],
) -> np.ndarray:
    initial_guess = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(
        objective,
        x0=initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=list(constraints),
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"optimization failed: {result.message}")
    return _normalize_weights(result.x)


def minimize_volatility(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: OptimizationConstraints | None = None,
) -> PortfolioSolution:
    """Find long-only minimum volatility portfolio under weight bounds."""
    constraints = constraints or OptimizationConstraints()
    _validate_inputs(expected_returns, covariance)

    assets = list(expected_returns.index)
    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    n_assets = len(assets)

    weight_bounds = _build_bounds(n_assets, constraints)
    eq_constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    weights = _run_optimization(
        objective=lambda w: np.sqrt(w @ sigma @ w),
        n_assets=n_assets,
        constraints=eq_constraints,
        bounds=weight_bounds,
    )

    portfolio_return, volatility, sharpe = _portfolio_stats(weights, mu, sigma, constraints.risk_free_rate)
    return PortfolioSolution(
        weights=pd.Series(weights, index=assets, name="weight"),
        expected_return=portfolio_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
    )


def maximize_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: OptimizationConstraints | None = None,
) -> PortfolioSolution:
    """Find portfolio with maximum Sharpe ratio under weight constraints."""
    constraints = constraints or OptimizationConstraints()
    _validate_inputs(expected_returns, covariance)

    assets = list(expected_returns.index)
    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    n_assets = len(assets)

    weight_bounds = _build_bounds(n_assets, constraints)
    eq_constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_return, portfolio_volatility, _ = _portfolio_stats(
            weights,
            mu,
            sigma,
            constraints.risk_free_rate,
        )
        if portfolio_volatility <= 1e-12:
            return 1e6
        return -((portfolio_return - constraints.risk_free_rate) / portfolio_volatility)

    weights = _run_optimization(
        objective=negative_sharpe,
        n_assets=n_assets,
        constraints=eq_constraints,
        bounds=weight_bounds,
    )

    portfolio_return, volatility, sharpe = _portfolio_stats(weights, mu, sigma, constraints.risk_free_rate)
    return PortfolioSolution(
        weights=pd.Series(weights, index=assets, name="weight"),
        expected_return=portfolio_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
    )


def build_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    num_points: int = 50,
    constraints: OptimizationConstraints | None = None,
) -> list[EfficientFrontierPoint]:
    """Build efficient frontier by solving minimum variance for target returns."""
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    constraints = constraints or OptimizationConstraints()
    _validate_inputs(expected_returns, covariance)

    assets = list(expected_returns.index)
    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    n_assets = len(assets)

    min_ret = float(np.min(mu))
    max_ret = float(np.max(mu))
    targets = np.linspace(min_ret, max_ret, num_points)

    weight_bounds = _build_bounds(n_assets, constraints)
    frontier: list[EfficientFrontierPoint] = []

    for target_return in targets:
        eq_constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_return: (w @ mu) - tr},
        )
        try:
            weights = _run_optimization(
                objective=lambda w: np.sqrt(w @ sigma @ w),
                n_assets=n_assets,
                constraints=eq_constraints,
                bounds=weight_bounds,
            )
            volatility = float(np.sqrt(weights @ sigma @ weights))
            frontier.append(
                EfficientFrontierPoint(
                    target_return=float(target_return),
                    volatility=volatility,
                    weights=pd.Series(weights, index=assets, name="weight"),
                )
            )
        except RuntimeError:
            continue

    if not frontier:
        raise RuntimeError("failed to build frontier for any target return")
    return frontier
