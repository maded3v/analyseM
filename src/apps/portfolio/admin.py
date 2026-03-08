from django.contrib import admin

from .models import Asset, OptimizationResult, Portfolio, PortfolioAsset, PriceHistory

admin.site.register(Portfolio)
admin.site.register(Asset)
admin.site.register(PortfolioAsset)
admin.site.register(PriceHistory)
admin.site.register(OptimizationResult)
