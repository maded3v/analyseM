from django.contrib import admin
from django.urls import include, path

from src.apps.portfolio.web_views import DashboardPageView, HowItWorksPageView, home_redirect

urlpatterns = [
    path("", home_redirect, name="home"),
    path("dashboard/", DashboardPageView.as_view(), name="dashboard-page"),
    path("dashboard/how-it-works/", HowItWorksPageView.as_view(), name="dashboard-how-it-works"),
    path("admin/", admin.site.urls),
    path("api/portfolio/", include("src.apps.portfolio.urls")),
]
