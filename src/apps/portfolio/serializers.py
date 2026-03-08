from rest_framework import serializers


class AnalysisRequestSerializer(serializers.Serializer):
    prices = serializers.DictField(
        child=serializers.ListField(child=serializers.FloatField()),
        allow_empty=False,
    )
    dates = serializers.ListField(child=serializers.CharField(), required=False)
    risk_free_rate = serializers.FloatField(required=False, default=0.0)
    min_weight = serializers.FloatField(required=False, default=0.0)
    max_weight = serializers.FloatField(required=False, default=1.0)
    frontier_points = serializers.IntegerField(required=False, default=50, min_value=2)
    n_portfolios = serializers.IntegerField(required=False, default=10000, min_value=100)
    dashboard_sample_limit = serializers.IntegerField(required=False, default=2000, min_value=50)
    seed = serializers.IntegerField(required=False, allow_null=True)
