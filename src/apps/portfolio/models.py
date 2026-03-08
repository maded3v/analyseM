from __future__ import annotations

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Asset(models.Model):
    ticker = models.CharField(max_length=16, unique=True)
    name = models.CharField(max_length=128, blank=True)

    def __str__(self) -> str:
        return self.ticker


class Portfolio(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="portfolios")
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    assets = models.ManyToManyField(Asset, through="PortfolioAsset", related_name="portfolios")

    def __str__(self) -> str:
        return f"{self.user_id}:{self.name}"


class PortfolioAsset(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    weight = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])

    class Meta:
        unique_together = ("portfolio", "asset")

    def __str__(self) -> str:
        return f"{self.portfolio_id}:{self.asset.ticker}={self.weight:.4f}"


class PriceHistory(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="price_history")
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    adjusted_close = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ("asset", "date")
        ordering = ["asset_id", "date"]


class OptimizationResult(models.Model):
    METHOD_MIN_VOL = "min_vol"
    METHOD_MAX_SHARPE = "max_sharpe"
    METHOD_MONTE_CARLO = "monte_carlo"

    METHOD_CHOICES = (
        (METHOD_MIN_VOL, "Minimum volatility"),
        (METHOD_MAX_SHARPE, "Maximum Sharpe"),
        (METHOD_MONTE_CARLO, "Monte Carlo"),
    )

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="optimization_results")
    method = models.CharField(max_length=32, choices=METHOD_CHOICES)
    weights = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
