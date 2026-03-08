from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go


def _figure_to_spec(fig: go.Figure) -> dict:
    """Вернуть JSON-совместимое представление графика Plotly."""
    return fig.to_plotly_json()


def build_correlation_heatmap(correlation: pd.DataFrame, title: str = "Матрица корреляций") -> dict:
    if correlation.empty:
        raise ValueError("correlation matrix is empty")

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=correlation.values,
                x=list(correlation.columns),
                y=list(correlation.index),
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar={"title": "corr"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Активы",
        yaxis_title="Активы",
        template="plotly_white",
    )
    return _figure_to_spec(fig)


def build_efficient_frontier_chart(
    frontier_points: Iterable[dict],
    min_volatility_point: dict,
    max_sharpe_point: dict,
    title: str = "Эффективная граница",
) -> dict:
    frontier_df = pd.DataFrame(frontier_points)
    if frontier_df.empty:
        raise ValueError("frontier_points is empty")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["target_return"],
            mode="lines+markers",
            name="Эффективная граница",
            marker={"size": 5},
            line={"width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_volatility_point["volatility"]],
            y=[min_volatility_point["expected_return"]],
            mode="markers",
            name="Минимальная волатильность",
            marker={"size": 12, "symbol": "diamond", "color": "green"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_point["volatility"]],
            y=[max_sharpe_point["expected_return"]],
            mode="markers",
            name="Максимальный Sharpe",
            marker={"size": 12, "symbol": "star", "color": "red"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Риск (sigma)",
        yaxis_title="Ожидаемая доходность",
        template="plotly_white",
    )
    return _figure_to_spec(fig)


def build_monte_carlo_chart(
    samples: pd.DataFrame,
    best_sharpe_point: dict,
    min_volatility_point: dict,
    title: str = "Симуляция портфелей Monte Carlo",
) -> dict:
    if samples.empty:
        raise ValueError("samples is empty")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=samples["volatility"],
            y=samples["expected_return"],
            mode="markers",
            name="Случайные портфели",
            marker={
                "size": 5,
                "color": samples["sharpe_ratio"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Sharpe"},
                "opacity": 0.7,
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_sharpe_point["volatility"]],
            y=[best_sharpe_point["expected_return"]],
            mode="markers",
            name="Лучший Sharpe",
            marker={"size": 13, "symbol": "star", "color": "red"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_volatility_point["volatility"]],
            y=[min_volatility_point["expected_return"]],
            mode="markers",
            name="Мин. волатильность",
            marker={"size": 13, "symbol": "diamond", "color": "green"},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Риск (sigma)",
        yaxis_title="Ожидаемая доходность",
        template="plotly_white",
    )
    return _figure_to_spec(fig)
