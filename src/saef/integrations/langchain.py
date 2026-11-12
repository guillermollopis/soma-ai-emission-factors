"""LangChain callback handler that estimates carbon and water for every finished LLM call.

    from saef.integrations.langchain import SaefCallbackHandler

    handler = SaefCallbackHandler(region="eu-west-1")
    llm.invoke("hello", config={"callbacks": [handler]})
    handler.totals()

For LangChain v1 agents, SaefMiddleware (needs the `langchain` package) records one estimate per model call:

    from langchain.agents import create_agent
    from saef.integrations.langchain import SaefMiddleware

    saef_mw = SaefMiddleware(region="eu-west-1", attach=True)
    agent = create_agent("openai:gpt-4o", tools=[], middleware=[saef_mw])
    saef_mw.totals()
"""
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from ._base import Tracker


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _usage(response):
    """(uncached_input, cached_input, output) from usage_metadata on the messages, else llm_output['token_usage']."""
    inp = cached = out = 0
    found = False
    for gens in response.generations:
        for g in gens:
            um = getattr(getattr(g, "message", None), "usage_metadata", None)
            if um:
                found = True
                c = _num((um.get("input_token_details") or {}).get("cache_read"))
                inp += _num(um.get("input_tokens")) - c
                cached += c
                out += _num(um.get("output_tokens"))
    if found:
        return inp, cached, out
    tu = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage") or {}
    prompt = _num(tu.get("prompt_tokens", tu.get("input_tokens")))
    c = _num((tu.get("prompt_tokens_details") or {}).get("cached_tokens"))
    return prompt - c, c, _num(tu.get("completion_tokens", tu.get("output_tokens")))


def _model_from_start(serialized, kwargs):
    inv = kwargs.get("invocation_params") or {}
    for k in ("model", "model_name", "model_id"):
        if inv.get(k):
            return inv[k]
    sk = (serialized or {}).get("kwargs") or {}
    for k in ("model", "model_name", "model_id"):
        if sk.get(k):
            return sk[k]
    return None


class SaefCallbackHandler(BaseCallbackHandler):
    """region defaults to env SAEF_REGION, then 'global-average'. The model name is read from the run's invocation
    parameters, then from llm_output['model_name'] or the message's response_metadata. Calls without a model name or
    without token usage are not counted."""

    def __init__(self, region=None, log_path=None):
        self._tracker = Tracker(region=region, log_path=log_path)
        self._models = {}

    def totals(self):
        return self._tracker.totals()

    def reset(self):
        self._tracker.reset()
        self._models.clear()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._models[run_id] = _model_from_start(serialized, kwargs)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._models[run_id] = _model_from_start(serialized, kwargs)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._models.pop(run_id, None)

    def on_llm_end(self, response, *, run_id, **kwargs):
        model = self._models.pop(run_id, None)
        lo = response.llm_output or {}
        model = model or lo.get("model_name") or lo.get("model")
        if not model:
            for gens in response.generations:
                for g in gens:
                    rm = getattr(getattr(g, "message", None), "response_metadata", None) or {}
                    model = model or rm.get("model_name") or rm.get("model")
        if not model:
            return
        inp, cached, out = _usage(response)
        if inp + cached + out <= 0:
            return
        self._tracker.record(model, inp, out, cached, extra={"run_id": str(run_id)})


try:
    from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse, ModelResponse
except ImportError:
    AgentMiddleware = None


def _ai_message(result):
    if isinstance(result, ExtendedModelResponse):
        result = result.model_response
    msgs = result.result if isinstance(result, ModelResponse) else [result]
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def _model_from_message(msg, request):
    rm = msg.response_metadata or {}
    model = rm.get("model_name") or rm.get("model")
    if model:
        return model
    m = getattr(request, "model", None)
    for k in ("model_name", "model", "model_id"):
        v = getattr(m, k, None)
        if isinstance(v, str) and v:
            return v
    return None


if AgentMiddleware is None:

    class SaefMiddleware:
        def __init__(self, *args, **kwargs):
            raise ImportError('SaefMiddleware needs LangChain v1: pip install langchain')

else:

    class SaefMiddleware(AgentMiddleware):
        """Estimates carbon and water for every model call an agent makes, via wrap_model_call and awrap_model_call.

        region defaults to env SAEF_REGION, then 'global-average'. The model name comes from the response metadata, then
        from the request's chat model. Calls without a model name or without usage_metadata are not counted. With
        attach=True the estimate is also written to AIMessage.response_metadata["saef"]."""

        def __init__(self, region=None, log_path=None, attach=False):
            super().__init__()
            self._tracker = Tracker(region=region, log_path=log_path)
            self.attach = attach

        def totals(self):
            return self._tracker.totals()

        def reset(self):
            self._tracker.reset()

        def _record(self, request, result):
            msg = _ai_message(result)
            if msg is None:
                return
            um = msg.usage_metadata
            if not um:
                return
            model = _model_from_message(msg, request)
            if not model:
                return
            c = _num((um.get("input_token_details") or {}).get("cache_read"))
            inp = _num(um.get("input_tokens")) - c
            out = _num(um.get("output_tokens"))
            if inp + c + out <= 0:
                return
            rec = self._tracker.record(model, inp, out, c, extra={"message_id": msg.id})
            if rec and self.attach:
                msg.response_metadata["saef"] = rec

        def wrap_model_call(self, request, handler):
            result = handler(request)
            self._record(request, result)
            return result

        async def awrap_model_call(self, request, handler):
            result = await handler(request)
            self._record(request, result)
            return result
