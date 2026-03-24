from django.shortcuts import redirect
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.generic import TemplateView


def home_redirect(_request):
    return redirect("dashboard-page")


@method_decorator(ensure_csrf_cookie, name="dispatch")
class DashboardPageView(TemplateView):
    template_name = "portfolio/dashboard.html"


class HowItWorksPageView(TemplateView):
    template_name = "portfolio/how_it_works.html"
