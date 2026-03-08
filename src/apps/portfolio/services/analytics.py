from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.portfolio_analysis.quant import (
    OptimizationConstraints,
    build_efficient_frontier,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_risk_report,
    maximize_sharpe,
    minimize_volatility,
    returns_from_prices,
    simulate_random_portfolios,
)
from src.portfolio_analysis.visualization import (
    build_correlation_heatmap,
    build_efficient_frontier_chart,
    build_monte_carlo_chart,
)


def dataframe_from_payload(payload: dict) -> pd.DataFrame:
    """Convert API payload into a price DataFrame.

    Expected schema:
    {
      "prices": {
         "AAPL": [100, 101, 102],
         "MSFT": [200, 198, 202]
      },
      "dates": ["2025-01-01", "2025-01-02", "2025-01-03"]
    }
    """
    prices = payload.get("prices", {})
    if not prices:
        raise ValueError("payload must contain non-empty 'prices'")

    dates = payload.get("dates")
    df = pd.DataFrame(prices)
    if dates:
        df.index = pd.to_datetime(dates)
    return df


def analyze_prices(payload: dict) -> dict:
    price_df = dataframe_from_payload(payload)
    returns = returns_from_prices(price_df)

    mean_returns = returns.mean() * 252
    covariance = compute_covariance_matrix(returns, annualize=True)
    correlation = compute_correlation_matrix(returns)

    risk_report = {
        asset: asdict(compute_risk_report(returns[asset]))
        for asset in returns.columns
    }

    return {
        "mean_returns": mean_returns.to_dict(),
        "covariance": covariance.to_dict(),
        "correlation": correlation.to_dict(),
        "risk_report": risk_report,
    }


def optimize_markowitz(payload: dict) -> dict:
    price_df = dataframe_from_payload(payload)
    returns = returns_from_prices(price_df)

    expected_returns = returns.mean() * 252
    covariance = compute_covariance_matrix(returns, annualize=True)

    constraints = OptimizationConstraints(
        min_weight=float(payload.get("min_weight", 0.0)),
        max_weight=float(payload.get("max_weight", 1.0)),
        risk_free_rate=float(payload.get("risk_free_rate", 0.0)),
    )

    min_vol_solution = minimize_volatility(expected_returns, covariance, constraints)
    max_sharpe_solution = maximize_sharpe(expected_returns, covariance, constraints)
    frontier = build_efficient_frontier(
        expected_returns,
        covariance,
        num_points=int(payload.get("frontier_points", 50)),
        constraints=constraints,
    )

    return {
        "min_volatility": {
            "weights": min_vol_solution.weights.to_dict(),
            "expected_return": min_vol_solution.expected_return,
            "volatility": min_vol_solution.volatility,
            "sharpe_ratio": min_vol_solution.sharpe_ratio,
        },
        "max_sharpe": {
            "weights": max_sharpe_solution.weights.to_dict(),
            "expected_return": max_sharpe_solution.expected_return,
            "volatility": max_sharpe_solution.volatility,
            "sharpe_ratio": max_sharpe_solution.sharpe_ratio,
        },
        "frontier": [
            {
                "target_return": point.target_return,
                "volatility": point.volatility,
                "weights": point.weights.to_dict(),
            }
            for point in frontier
        ],
    }


def run_monte_carlo(payload: dict) -> dict:
    price_df = dataframe_from_payload(payload)
    returns = returns_from_prices(price_df)

    expected_returns = returns.mean() * 252
    covariance = compute_covariance_matrix(returns, annualize=True)

    simulation = simulate_random_portfolios(
        expected_returns,
        covariance,
        n_portfolios=int(payload.get("n_portfolios", 10_000)),
        risk_free_rate=float(payload.get("risk_free_rate", 0.0)),
        seed=payload.get("seed"),
    )

    return {
        "best_sharpe": simulation.best_sharpe.to_dict(),
        "min_volatility": simulation.min_volatility.to_dict(),
        "samples": simulation.samples.to_dict(orient="records"),
    }


def build_dashboard(payload: dict) -> dict:
    """Build complete analysis output with Plotly-ready charts."""
    price_df = dataframe_from_payload(payload)
    returns = returns_from_prices(price_df)

    expected_returns = returns.mean() * 252
    covariance = compute_covariance_matrix(returns, annualize=True)
    correlation = compute_correlation_matrix(returns)

    constraints = OptimizationConstraints(
        min_weight=float(payload.get("min_weight", 0.0)),
        max_weight=float(payload.get("max_weight", 1.0)),
        risk_free_rate=float(payload.get("risk_free_rate", 0.0)),
    )

    min_vol_solution = minimize_volatility(expected_returns, covariance, constraints)
    max_sharpe_solution = maximize_sharpe(expected_returns, covariance, constraints)
    frontier = build_efficient_frontier(
        expected_returns,
        covariance,
        num_points=int(payload.get("frontier_points", 50)),
        constraints=constraints,
    )
    simulation = simulate_random_portfolios(
        expected_returns,
        covariance,
        n_portfolios=int(payload.get("n_portfolios", 10_000)),
        risk_free_rate=constraints.risk_free_rate,
        seed=payload.get("seed"),
    )

    min_vol_payload = {
        "weights": min_vol_solution.weights.to_dict(),
        "expected_return": min_vol_solution.expected_return,
        "volatility": min_vol_solution.volatility,
        "sharpe_ratio": min_vol_solution.sharpe_ratio,
    }
    max_sharpe_payload = {
        "weights": max_sharpe_solution.weights.to_dict(),
        "expected_return": max_sharpe_solution.expected_return,
        "volatility": max_sharpe_solution.volatility,
        "sharpe_ratio": max_sharpe_solution.sharpe_ratio,
    }
    frontier_payload = [
        {
            "target_return": point.target_return,
            "volatility": point.volatility,
            "weights": point.weights.to_dict(),
        }
        for point in frontier
    ]

    sample_limit = int(payload.get("dashboard_sample_limit", 2000))
    monte_carlo_samples = simulation.samples.head(sample_limit).copy()

    charts = {
        "correlation_heatmap": build_correlation_heatmap(correlation),
        "efficient_frontier": build_efficient_frontier_chart(
            frontier_points=frontier_payload,
            min_volatility_point=min_vol_payload,
            max_sharpe_point=max_sharpe_payload,
        ),
        "monte_carlo": build_monte_carlo_chart(
            samples=monte_carlo_samples,
            best_sharpe_point=simulation.best_sharpe.to_dict(),
            min_volatility_point=simulation.min_volatility.to_dict(),
        ),
    }

    return {
        "summary": {
            "mean_returns": expected_returns.to_dict(),
            "covariance": covariance.to_dict(),
            "correlation": correlation.to_dict(),
            "risk_report": {
                asset: asdict(compute_risk_report(returns[asset]))
                for asset in returns.columns
            },
        },
        "optimization": {
            "min_volatility": min_vol_payload,
            "max_sharpe": max_sharpe_payload,
            "frontier": frontier_payload,
        },
        "monte_carlo": {
            "best_sharpe": simulation.best_sharpe.to_dict(),
            "min_volatility": simulation.min_volatility.to_dict(),
            "samples": monte_carlo_samples.to_dict(orient="records"),
            "total_generated": int(simulation.samples.shape[0]),
            "returned_samples": int(monte_carlo_samples.shape[0]),
        },
        "charts": charts,
    }
