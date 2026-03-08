from django.urls import path

from .views import AnalyzeView, ApiIndexView, DashboardView, MonteCarloView, OptimizeMarkowitzView

urlpatterns = [
    path("", ApiIndexView.as_view(), name="portfolio-api-index"),
    path("analyze/", AnalyzeView.as_view(), name="portfolio-analyze"),
    path("optimize/markowitz/", OptimizeMarkowitzView.as_view(), name="portfolio-optimize-markowitz"),
    path("simulate/monte-carlo/", MonteCarloView.as_view(), name="portfolio-simulate-monte-carlo"),
    path("dashboard/", DashboardView.as_view(), name="portfolio-dashboard"),
]
