"""LiteLLM callback that estimates carbon and water for every successful call.

    import litellm
    from saef.integrations.litellm import SaefLogger

    logger = SaefLogger(region="eu-west-1")
    litellm.callbacks = [logger]
    ...
    logger.totals()
"""
from litellm.integrations.custom_logger import CustomLogger

from ._base import Tracker
from .openai import usage_from_response

REGION_PARAMS = {
    "bedrock": "aws_region_name",
    "sagemaker": "aws_region_name",
    "vertex_ai": "vertex_location",
    "vertex_ai_beta": "vertex_location",
}


class SaefLogger(CustomLogger):
    """region defaults to env SAEF_REGION, then 'global-average'. For Bedrock, SageMaker and Vertex AI the call's own
    region is used when LiteLLM exposes it. region_map, for example {"azure": "swedencentral"}, overrides the
    default per LiteLLM provider. The estimate is attached to response._hidden_params["saef"]."""

    def __init__(self, region=None, region_map=None, log_path=None, **kwargs):
        super().__init__(**kwargs)
        self._tracker = Tracker(region=region, log_path=log_path)
        self.region_map = dict(region_map or {})

    def totals(self):
        return self._tracker.totals()

    def reset(self):
        self._tracker.reset()

    def _region_for(self, kwargs):
        params = kwargs.get("litellm_params") or {}
        provider = params.get("custom_llm_provider") or kwargs.get("custom_llm_provider")
        if provider in self.region_map:
            return self.region_map[provider], provider
        key = REGION_PARAMS.get(provider)
        call_region = (params.get(key) or (kwargs.get("optional_params") or {}).get(key)) if key else None
        return call_region, provider

    def _handle(self, kwargs, response_obj):
        try:
            model, inp, cached, out = usage_from_response(response_obj)
        except Exception:
            std = kwargs.get("standard_logging_object") or {}
            if not std.get("prompt_tokens") and not std.get("completion_tokens"):
                return
            model, inp, cached, out = None, std.get("prompt_tokens") or 0, 0, std.get("completion_tokens") or 0
        model = kwargs.get("model") or model
        if not model:
            return
        region, provider = self._region_for(kwargs)
        res = self._tracker.record(
            model, inp, out, cached, region=region,
            extra={"call_id": kwargs.get("litellm_call_id"), "provider": provider},
        )
        if res is not None:
            hidden = getattr(response_obj, "_hidden_params", None)
            if isinstance(hidden, dict):
                hidden["saef"] = res

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle(kwargs, response_obj)
