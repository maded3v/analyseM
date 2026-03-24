from typing import Any

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import AnalysisRequestSerializer
from .services.analytics import analyze_prices, build_dashboard, optimize_markowitz, run_monte_carlo
from .services.moex import fetch_price_history, search_tickers


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
                    "moex_suggest": "/api/portfolio/moex/suggest/?q=SBER",
                    "moex_history": "/api/portfolio/moex/history/?secid=SBER",
                },
                "method": "Для аналитических endpoint используйте POST",
            },
            status=status.HTTP_200_OK,
        )


class AnalyzeView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload: dict[str, Any] = dict(serializer.validated_data)
        result = analyze_prices(payload)
        return Response(result, status=status.HTTP_200_OK)


class OptimizeMarkowitzView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload: dict[str, Any] = dict(serializer.validated_data)
        result = optimize_markowitz(payload)
        return Response(result, status=status.HTTP_200_OK)


class MonteCarloView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload: dict[str, Any] = dict(serializer.validated_data)
        result = run_monte_carlo(payload)
        return Response(result, status=status.HTTP_200_OK)


class DashboardView(APIView):
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload: dict[str, Any] = dict(serializer.validated_data)
        result = build_dashboard(payload)
        return Response(result, status=status.HTTP_200_OK)


class MoexSuggestView(APIView):
    def get(self, request):
        query = str(request.query_params.get("q", "")).strip()
        suggestions = search_tickers(query)
        return Response({"query": query, "suggestions": suggestions}, status=status.HTTP_200_OK)


class MoexHistoryView(APIView):
    def get(self, request):
        secid = str(request.query_params.get("secid", "")).strip()
        if not secid:
            return Response({"detail": "Параметр secid обязателен"}, status=status.HTTP_400_BAD_REQUEST)

        board = str(request.query_params.get("board", "TQBR")).strip() or "TQBR"
        from_date = request.query_params.get("from")
        till_date = request.query_params.get("till")

        try:
            result = fetch_price_history(secid=secid, board=board, from_date=from_date, till_date=till_date)
        except ValueError as error:
            return Response({"detail": str(error)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            return Response(
                {"detail": "Не удалось получить данные с MOEX. Повторите позже."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response(
            {
                "source": "moex",
                "secid": result.secid,
                "board": result.board,
                "dates": result.dates,
                "prices": result.prices,
                "avg_price": result.avg_price,
                "points": len(result.prices),
            },
            status=status.HTTP_200_OK,
        )
