# Архитектура сервиса анализа портфеля

## Структура проекта

```text
analiseM/
  manage.py
  config/
    settings.py
    urls.py
    asgi.py
    wsgi.py
  src/
    apps/
      portfolio/
        models.py
        serializers.py
        views.py
        urls.py
        services/
          analytics.py
    portfolio_analysis/
      quant/
        markowitz.py
        monte_carlo.py
        correlation.py
        risk_metrics.py
      visualization/
        plotly_charts.py
```

## Сущности предметной области

- User (встроенная модель Django)
- Portfolio
- Asset
- PriceHistory
- OptimizationResult

## Аналитический слой

- `correlation.py`: расчет доходностей, ковариационной и корреляционной матриц.
- `risk_metrics.py`: годовая доходность, волатильность, Sharpe, Sortino, VaR, CVaR, максимальная просадка.
- `markowitz.py`: оптимизация минимальной волатильности, максимального Sharpe и построение эффективной границы.
- `monte_carlo.py`: генерация случайных портфелей и выбор лучших кандидатов.
- `plotly_charts.py`: подготовка JSON-спецификаций графиков Plotly для фронтенда.

## API endpoints

- `POST /api/portfolio/analyze/`
- `POST /api/portfolio/optimize/markowitz/`
- `POST /api/portfolio/simulate/monte-carlo/`
- `POST /api/portfolio/dashboard/`

Базовый формат запроса:

```json
{
  "prices": {
    "AAPL": [100, 101, 102],
    "MSFT": [200, 199, 201]
  },
  "dates": ["2025-01-01", "2025-01-02", "2025-01-03"]
}
```

Дополнительные параметры:

- `risk_free_rate` (float)
- `min_weight` и `max_weight` (float)
- `frontier_points` (int)
- `n_portfolios` (int)
- `dashboard_sample_limit` (int)
- `seed` (int)

## Математическая модель

- Ожидаемая доходность портфеля: `R_p = w^T * mu`
- Риск (волатильность) портфеля: `sigma_p = sqrt(w^T * Sigma * w)`
- Коэффициент Sharpe: `S = (R_p - R_f) / sigma_p`

Ограничения оптимизации:

- `sum(w_i) = 1`
- `min_weight <= w_i <= max_weight`
- По умолчанию long-only: `w_i >= 0`
