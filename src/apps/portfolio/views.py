from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import AnalysisRequestSerializer
from .services.analytics import analyze_prices, build_dashboard, optimize_markowitz, run_monte_carlo


class ApiIndexView(APIView):
    """Индекс API для локальной разработки."""

    def get(self, request):
        return Response(
            {
                "service": "Сервис анализа инвестиционного портфеля",
                "endpoints": {
                    "analyze": "/api/portfolio/analyze/",
                    "optimize_markowitz": "/api/portfolio/optimize/markowitz/",
                    "simulate_monte_carlo": "/api/portfolio/simulate/monte-carlo/",
                    "dashboard": "/api/portfolio/dashboard/",
                },
                "method": "Для аналитических endpoint используйте POST",
            },
            status=status.HTTP_200_OK,
        )


class AnalyzeView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = analyze_prices(serializer.validated_data)
        return Response(result, status=status.HTTP_200_OK)


class OptimizeMarkowitzView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = optimize_markowitz(serializer.validated_data)
        return Response(result, status=status.HTTP_200_OK)


class MonteCarloView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = run_monte_carlo(serializer.validated_data)
        return Response(result, status=status.HTTP_200_OK)


class DashboardView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = build_dashboard(serializer.validated_data)
        return Response(result, status=status.HTTP_200_OK)
