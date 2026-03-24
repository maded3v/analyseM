from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
from typing import Any

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
    simulate_portfolio_terminal_values,
    simulate_random_portfolios,
)
from src.portfolio_analysis.visualization import (
    build_correlation_heatmap,
    build_efficient_frontier_chart,
    build_monte_carlo_chart,
    build_projection_distribution_chart,
)


def dataframe_from_payload(payload: dict) -> pd.DataFrame:
    prices = payload.get("prices", {})
    if not prices:
        raise ValueError("payload must contain non-empty 'prices'")

    dates = payload.get("dates")
    df = pd.DataFrame(prices)
    if dates:
        df.index = pd.to_datetime(dates)
    return df


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _annualized_mean_returns(returns: pd.DataFrame) -> pd.Series:
    values = pd.Series(returns.mean(numeric_only=True), dtype=float)
    return values * 252.0


def _normalize_weights(weights_payload: dict | None, assets: list[str]) -> pd.Series:
    if not assets:
        raise ValueError("assets list is empty")

    if not weights_payload:
        equal = 1.0 / len(assets)
        return pd.Series({asset: equal for asset in assets}, dtype=float)

    weights = pd.Series(weights_payload, dtype=float).reindex(assets).fillna(0.0)
    if (weights < 0).any():
        raise ValueError("current_weights must be non-negative")
    total = float(weights.sum())
    if total <= 1e-12:
        raise ValueError("current_weights sum must be > 0")
    return weights / total


def _portfolio_metrics(
    returns: pd.DataFrame,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float,
) -> dict:
    aligned = weights.reindex(expected_returns.index).fillna(0.0)
    aligned = aligned / float(aligned.sum())

    expected_return = float(aligned @ expected_returns)
    volatility = float((aligned.T @ covariance @ aligned) ** 0.5)
    sharpe_ratio = 0.0 if volatility <= 1e-12 else (expected_return - risk_free_rate) / volatility

    portfolio_daily_returns = returns @ aligned
    report = compute_risk_report(portfolio_daily_returns, risk_free_rate=risk_free_rate)

    return {
        "weights": aligned.to_dict(),
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "risk": asdict(report),
    }


def _pair_correlation_insights(correlation: pd.DataFrame) -> dict:
    assets = list(correlation.columns)
    if len(assets) < 2:
        return {"max_positive": None, "max_negative": None}

    pairs: list[dict] = []
    for a, b in combinations(assets, 2):
        pairs.append({"asset_a": a, "asset_b": b, "value": float(correlation.loc[a, b])})

    max_positive = max(pairs, key=lambda item: item["value"])
    max_negative = min(pairs, key=lambda item: item["value"])
    return {"max_positive": max_positive, "max_negative": max_negative}


def _diversification_score(weights: pd.Series) -> float:
    hhi = float((weights**2).sum())
    n_assets = max(int(weights.shape[0]), 1)
    normalized = (hhi - (1.0 / n_assets)) / (1.0 - (1.0 / n_assets)) if n_assets > 1 else 1.0
    return float(max(0.0, 1.0 - normalized))


def _profile_target_weights(min_vol_weights: pd.Series, max_sharpe_weights: pd.Series, profile: str) -> pd.Series:
    if profile == "conservative":
        return min_vol_weights.copy()
    if profile == "aggressive":
        return max_sharpe_weights.copy()
    return 0.5 * min_vol_weights + 0.5 * max_sharpe_weights


def _build_rebalancing_plan(
    current_weights: pd.Series,
    target_weights: pd.Series,
    threshold: float,
) -> list[dict]:
    actions: list[dict] = []
    for asset in target_weights.index:
        current_raw = current_weights.get(asset, 0.0)
        target_raw = target_weights.get(asset, 0.0)
        current = _safe_float(current_raw, 0.0)
        target = _safe_float(target_raw, 0.0)
        delta = target - current

        if abs(delta) < 1e-12:
            action = "hold"
        elif delta > threshold:
            action = "buy"
        elif delta < -threshold:
            action = "sell"
        else:
            action = "adjust_minor"

        actions.append(
            {
                "asset": asset,
                "current_weight": current,
                "target_weight": target,
                "delta": delta,
                "action": action,
                "priority": "high" if abs(delta) >= threshold else "low",
            }
        )

    actions.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return actions


def analyze_prices(payload: dict) -> dict:
    price_df = dataframe_from_payload(payload)
    returns: pd.DataFrame = returns_from_prices(price_df)

    mean_returns: pd.Series = _annualized_mean_returns(returns)
    covariance: pd.DataFrame = compute_covariance_matrix(returns, annualize=True)
    correlation: pd.DataFrame = compute_correlation_matrix(returns)

    risk_report = {
        asset: asdict(compute_risk_report(pd.Series(returns.loc[:, asset], dtype=float)))
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
    returns: pd.DataFrame = returns_from_prices(price_df)

    expected_returns: pd.Series = _annualized_mean_returns(returns)
    covariance: pd.DataFrame = compute_covariance_matrix(returns, annualize=True)

    constraints = OptimizationConstraints(
        min_weight=float(payload.get("min_weight", 0.0)),
        max_weight=float(payload.get("max_weight", 1.0)),
        risk_free_rate=float(payload.get("risk_free_rate", 0.125)),
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
    returns: pd.DataFrame = returns_from_prices(price_df)

    expected_returns: pd.Series = _annualized_mean_returns(returns)
    covariance: pd.DataFrame = compute_covariance_matrix(returns, annualize=True)

    simulation = simulate_random_portfolios(
        expected_returns,
        covariance,
        n_portfolios=int(payload.get("n_portfolios", 10_000)),
        risk_free_rate=float(payload.get("risk_free_rate", 0.125)),
        seed=payload.get("seed"),
    )

    return {
        "best_sharpe": simulation.best_sharpe.to_dict(),
        "min_volatility": simulation.min_volatility.to_dict(),
        "samples": simulation.samples.to_dict(orient="records"),
    }


def build_dashboard(payload: dict) -> dict:
    price_df = dataframe_from_payload(payload)
    returns: pd.DataFrame = returns_from_prices(price_df)

    expected_returns: pd.Series = _annualized_mean_returns(returns)
    covariance: pd.DataFrame = compute_covariance_matrix(returns, annualize=True)
    correlation: pd.DataFrame = compute_correlation_matrix(returns)

    risk_free_rate = _safe_float(payload.get("risk_free_rate"), 0.125)
    constraints = OptimizationConstraints(
        min_weight=_safe_float(payload.get("min_weight"), 0.0),
        max_weight=_safe_float(payload.get("max_weight"), 1.0),
        risk_free_rate=risk_free_rate,
    )

    current_weights = _normalize_weights(payload.get("current_weights"), list(expected_returns.index))

    min_vol_solution = minimize_volatility(expected_returns, covariance, constraints)
    max_sharpe_solution = maximize_sharpe(expected_returns, covariance, constraints)
    frontier = build_efficient_frontier(
        expected_returns,
        covariance,
        num_points=_safe_int(payload.get("frontier_points"), 50),
        constraints=constraints,
    )
    simulation = simulate_random_portfolios(
        expected_returns,
        covariance,
        n_portfolios=_safe_int(payload.get("n_portfolios"), 10_000),
        risk_free_rate=risk_free_rate,
        seed=payload.get("seed"),
    )

    min_vol_weights = min_vol_solution.weights.reindex(expected_returns.index).fillna(0.0)
    max_sharpe_weights = max_sharpe_solution.weights.reindex(expected_returns.index).fillna(0.0)

    profile = str(payload.get("target_profile", "balanced"))
    target_weights = _profile_target_weights(min_vol_weights, max_sharpe_weights, profile)
    target_weights = target_weights / float(target_weights.sum())

    current_portfolio = _portfolio_metrics(
        returns=returns,
        expected_returns=expected_returns,
        covariance=covariance,
        weights=current_weights,
        risk_free_rate=risk_free_rate,
    )
    target_portfolio = _portfolio_metrics(
        returns=returns,
        expected_returns=expected_returns,
        covariance=covariance,
        weights=target_weights,
        risk_free_rate=risk_free_rate,
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

    sample_limit = _safe_int(payload.get("dashboard_sample_limit"), 2000)
    monte_carlo_samples = simulation.samples.head(sample_limit).copy()

    rebalancing_threshold = _safe_float(payload.get("rebalancing_threshold"), 0.03)
    rebalancing_plan = _build_rebalancing_plan(current_weights, target_weights, rebalancing_threshold)

    horizon_days = _safe_int(payload.get("mc_horizon_days"), 252)
    n_paths = _safe_int(payload.get("mc_paths"), 3000)
    projection = simulate_portfolio_terminal_values(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=target_weights,
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=payload.get("seed"),
    )

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
        "projection_distribution": build_projection_distribution_chart(
            terminal_values=projection.terminal_values,
            quantiles=projection.quantiles,
        ),
    }

    correlation_insights = _pair_correlation_insights(correlation)

    return {
        "summary": {
            "mean_returns": expected_returns.to_dict(),
            "covariance": covariance.to_dict(),
            "correlation": correlation.to_dict(),
            "risk_report": {
                asset: asdict(compute_risk_report(pd.Series(returns.loc[:, asset], dtype=float)))
                for asset in returns.columns
            },
            "current_weights": current_weights.to_dict(),
            "diversification_score": _diversification_score(current_weights),
            "correlation_insights": correlation_insights,
        },
        "optimization": {
            "min_volatility": min_vol_payload,
            "max_sharpe": max_sharpe_payload,
            "frontier": frontier_payload,
            "target_profile": profile,
            "target_weights": target_weights.to_dict(),
        },
        "risk_management": {
            "current_portfolio": current_portfolio,
            "target_portfolio": target_portfolio,
            "rebalancing_threshold": rebalancing_threshold,
            "rebalancing_plan": rebalancing_plan,
        },
        "projection": {
            "horizon_days": horizon_days,
            "n_paths": n_paths,
            "mean_terminal": projection.mean_terminal,
            "median_terminal": projection.median_terminal,
            "quantiles": projection.quantiles,
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
