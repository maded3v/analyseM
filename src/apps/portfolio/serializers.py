from rest_framework import serializers


class AnalysisRequestSerializer(serializers.Serializer):
    prices = serializers.DictField(
        child=serializers.ListField(child=serializers.FloatField()),
        allow_empty=False,
    )
    dates = serializers.ListField(child=serializers.CharField(), required=False)
    current_weights = serializers.DictField(child=serializers.FloatField(), required=False)
    risk_free_rate = serializers.FloatField(required=False, default=0.125)
    min_weight = serializers.FloatField(required=False, default=0.0)
    max_weight = serializers.FloatField(required=False, default=1.0)
    rebalancing_threshold = serializers.FloatField(required=False, default=0.03, min_value=0.0, max_value=1.0)
    target_profile = serializers.ChoiceField(
        required=False,
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
    )
    frontier_points = serializers.IntegerField(required=False, default=50, min_value=2)
    n_portfolios = serializers.IntegerField(required=False, default=10000, min_value=100)
    mc_horizon_days = serializers.IntegerField(required=False, default=252, min_value=20, max_value=2520)
    mc_paths = serializers.IntegerField(required=False, default=3000, min_value=500, max_value=50000)
    dashboard_sample_limit = serializers.IntegerField(required=False, default=2000, min_value=50)
    seed = serializers.IntegerField(required=False, allow_null=True)
