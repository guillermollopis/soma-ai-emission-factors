"""LlamaIndex event handler that estimates carbon and water for every finished LLM call.

    from llama_index.core.instrumentation import get_dispatcher
    from saef.integrations.llamaindex import SaefEventHandler

    handler = SaefEventHandler(region="eu-west-1")
    get_dispatcher().add_event_handler(handler)
    ...
    handler.totals()

SaefEventHandler.register() does the add_event_handler call for you.
"""
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from pydantic import PrivateAttr

from .. import SaefError
from ._base import Tracker
from .openai import usage_from_response


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _model_name(model_dict):
    d = model_dict or {}
    for k in ("model", "model_name", "model_id"):
        if isinstance(d.get(k), str) and d[k]:
            return d[k]
    return None


def _usage(response):
    """(model, uncached_input, cached_input, output) from response.raw, else response.additional_kwargs."""
    raw = getattr(response, "raw", None)
    if raw is not None:
        try:
            return usage_from_response(raw)
        except SaefError:
            pass
    ak = getattr(response, "additional_kwargs", None) or {}
    prompt = _num(ak.get("prompt_tokens", ak.get("input_tokens")))
    return None, prompt, 0, _num(ak.get("completion_tokens", ak.get("output_tokens")))


class SaefEventHandler(BaseEventHandler):
    """region defaults to env SAEF_REGION, then 'global-average'. The model name is read from the raw provider
    response, then from the model recorded by the matching LLM start event. Calls without a model name or without
    token usage are not counted. Streaming calls are counted only if the last chunk carries usage."""

    _tracker: Tracker = PrivateAttr()
    _models: dict = PrivateAttr(default_factory=dict)

    def __init__(self, region=None, log_path=None, **kwargs):
        super().__init__(**kwargs)
        self._tracker = Tracker(region=region, log_path=log_path)

    @classmethod
    def class_name(cls):
        return "SaefEventHandler"

    def register(self, dispatcher=None):
        (dispatcher or get_dispatcher()).add_event_handler(self)
        return self

    def totals(self):
        return self._tracker.totals()

    def reset(self):
        self._tracker.reset()
        self._models.clear()

    def handle(self, event, **kwargs):
        if isinstance(event, (LLMChatStartEvent, LLMCompletionStartEvent)):
            self._models[event.span_id] = _model_name(event.model_dict)
        elif isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
            self._record(event)

    def _record(self, event):
        started = self._models.pop(event.span_id, None)
        if event.response is None:
            return
        model, inp, cached, out = _usage(event.response)
        model = model or started
        if not model or inp + cached + out <= 0:
            return
        self._tracker.record(model, inp, out, cached, extra={"span_id": event.span_id})
