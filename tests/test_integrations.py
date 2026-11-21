import json
import uuid
from types import SimpleNamespace as NS

import pytest

import saef
from saef.integrations._base import Tracker
from saef.integrations.openai import estimate_from_response, usage_from_response


def direct(**kw):
    return saef.estimate(**kw)


def test_openai_chat_with_cache():
    resp = NS(model="gpt-4o", usage=NS(prompt_tokens=1000, completion_tokens=200, prompt_tokens_details=NS(cached_tokens=400)))
    assert usage_from_response(resp) == ("gpt-4o", 600, 400, 200)
    e = estimate_from_response(resp, region="eu-west-1")
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=600, cached_input_tokens=400, output_tokens=200)
    assert e.kg_co2e == d.kg_co2e and e.region_id == "eu-west-1"


def test_openai_chat_dict_without_details():
    resp = {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": None}}
    assert usage_from_response(resp) == ("gpt-4o-mini", 10, 0, 5)


def test_openai_responses_api():
    resp = NS(object="response", model="gpt-4o", usage=NS(input_tokens=500, output_tokens=50, input_tokens_details=NS(cached_tokens=100)))
    assert usage_from_response(resp) == ("gpt-4o", 400, 100, 50)


def test_anthropic_messages_cache_read_and_write():
    resp = NS(type="message", model="claude-3-5-sonnet-20241022",
              usage=NS(input_tokens=100, output_tokens=40, cache_read_input_tokens=300, cache_creation_input_tokens=50))
    assert usage_from_response(resp) == ("claude-3-5-sonnet-20241022", 150, 300, 40)
    assert estimate_from_response(resp).kg_co2e > 0


def test_anthropic_cache_fields_none():
    resp = NS(model="claude-3-5-haiku", usage=NS(input_tokens=10, output_tokens=2, cache_read_input_tokens=None, cache_creation_input_tokens=None))
    assert usage_from_response(resp) == ("claude-3-5-haiku", 10, 0, 2)


def test_missing_usage_and_model():
    with pytest.raises(saef.SaefError):
        estimate_from_response(NS(model="gpt-4o", usage=None))
    with pytest.raises(saef.SaefError):
        estimate_from_response(NS(usage=NS(prompt_tokens=1, completion_tokens=1)))
    assert estimate_from_response(NS(usage=NS(prompt_tokens=1, completion_tokens=1)), model="gpt-4o").kg_co2e > 0


def test_default_region_from_env(monkeypatch):
    monkeypatch.setenv("SAEF_REGION", "eu-west-1")
    assert Tracker().region == "eu-west-1"
    monkeypatch.delenv("SAEF_REGION")
    assert Tracker().region == "global-average"


def test_tracker_totals_log_and_fallbacks(tmp_path):
    log = tmp_path / "saef.jsonl"
    t = Tracker(region="eu-west-1", log_path=str(log))
    a = t.record("gpt-4o", 100, 50)
    t.record("gpt-4o", 100, 50, region="not-a-region")
    assert t.record("text-embedding-3-small", 100, 0) is None
    tot = t.totals()
    assert tot["calls"] == 2 and tot["skipped_calls"] == 1
    assert tot["kg_co2e"] == pytest.approx(2 * a["kg_co2e"])
    assert tot["by_model"]["gpt-4o"]["calls"] == 2
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["kg_co2e"] == a["kg_co2e"]
    t.reset()
    assert t.totals()["calls"] == 0


def test_tracker_cost_and_intensity():
    t = Tracker(region="eu-west-1")
    a = t.record("gpt-4o", 100, 50, cost=0.002)
    b = t.record("gpt-4o", 100, 50)
    assert a["cost"] == 0.002 and a["cost_currency"] == "USD"
    assert "cost" not in b and "cost_currency" not in b
    tot = t.totals()
    assert tot["cost"] == pytest.approx(0.002) and tot["calls_with_cost"] == 1
    assert tot["g_co2e_per_usd"] == pytest.approx(a["kg_co2e"] * 1000 / 0.002)


def test_tracker_without_cost_has_no_cost_keys():
    t = Tracker(region="eu-west-1")
    t.record("gpt-4o", 100, 50)
    tot = t.totals()
    assert "cost" not in tot and "g_co2e_per_usd" not in tot


def test_tracker_zero_cost_no_division_error():
    t = Tracker(region="eu-west-1")
    r = t.record("gpt-4o", 100, 50, cost=0.0)
    assert r["cost"] == 0.0
    tot = t.totals()
    assert tot["cost"] == 0.0 and tot["g_co2e_per_usd"] is None


def test_tracker_ignores_invalid_cost_and_mixed_currency():
    t = Tracker(region="eu-west-1")
    for bad in (-1, "0.1", True, float("nan")):
        assert "cost" not in t.record("gpt-4o", 10, 5, cost=bad)
    t.record("gpt-4o", 10, 5, cost=1.0)
    t.record("gpt-4o", 10, 5, cost=1.0, cost_currency="EUR")
    tot = t.totals()
    assert "cost" not in tot and tot["cost_by_currency"] == {"USD": 1.0, "EUR": 1.0}


def test_litellm_logger_cost():
    pytest.importorskip("litellm")
    from saef.integrations.litellm import SaefLogger

    lg = SaefLogger(region="eu-west-1")
    usage = NS(prompt_tokens=100, completion_tokens=10)
    r1 = NS(model="gpt-4o", usage=usage, _hidden_params={})
    lg.log_success_event({"model": "gpt-4o", "response_cost": 0.004}, r1, None, None)
    assert r1._hidden_params["saef"]["cost"] == 0.004
    r2 = NS(model="gpt-4o", usage=usage, _hidden_params={})
    kw = {"model": "gpt-4o", "standard_logging_object": {"response_cost": 0.006}}
    lg.log_success_event(kw, r2, None, None)
    assert r2._hidden_params["saef"]["cost_currency"] == "USD"
    tot = lg.totals()
    assert tot["cost"] == pytest.approx(0.01) and tot["calls_with_cost"] == 2
    assert tot["g_co2e_per_usd"] == pytest.approx(tot["kg_co2e"] * 1000 / 0.01)


def test_litellm_logger_no_cost_and_zero_cost():
    pytest.importorskip("litellm")
    from saef.integrations.litellm import SaefLogger

    lg = SaefLogger(region="eu-west-1")
    usage = NS(prompt_tokens=100, completion_tokens=10)
    r1 = NS(model="gpt-4o", usage=usage, _hidden_params={})
    lg.log_success_event({"model": "gpt-4o", "response_cost": None}, r1, None, None)
    assert "cost" not in r1._hidden_params["saef"] and "cost" not in lg.totals()
    r2 = NS(model="gpt-4o", usage=usage, _hidden_params={})
    lg.log_success_event({"model": "gpt-4o", "response_cost": 0.0}, r2, None, None)
    assert r2._hidden_params["saef"]["cost"] == 0.0
    assert lg.totals()["g_co2e_per_usd"] is None


def test_summarize():
    t = Tracker(region="eu-west-1")
    recs = [t.record("gpt-4o", 100, 50, cost=0.002), t.record("gpt-4o", 100, 50, cost=0.002), t.record("gpt-4o", 100, 50)]
    s = saef.summarize(recs)
    assert s["calls"] == 3 and s["kg_co2e"] == pytest.approx(sum(r["kg_co2e"] for r in recs))
    assert s["cost"] == pytest.approx(0.004) and s["calls_with_cost"] == 2
    assert s["g_co2e_per_usd"] == pytest.approx(2 * recs[0]["kg_co2e"] * 1000 / 0.004)
    assert "kwh" not in s


def test_summarize_without_cost_zero_cost_and_estimates():
    e = saef.estimate(model="gpt-4o", region="eu-west-1", input_tokens=100, output_tokens=50)
    s = saef.summarize([e, e.to_dict()])
    assert s["calls"] == 2 and "cost" not in s and "g_co2e_per_usd" not in s
    z = saef.summarize([{"kg_co2e": 1e-3, "cost": 0.0, "kwh": 0.5}])
    assert z["cost"] == 0.0 and z["g_co2e_per_usd"] is None and z["kwh"] == 0.5
    assert saef.summarize([])["calls"] == 0


def test_litellm_logger(tmp_path):
    pytest.importorskip("litellm")
    from saef.integrations.litellm import SaefLogger

    log = tmp_path / "l.jsonl"
    lg = SaefLogger(region="eu-west-1", region_map={"azure": "swedencentral"}, log_path=str(log))
    resp = NS(model="gpt-4o", usage=NS(prompt_tokens=1000, completion_tokens=200, prompt_tokens_details=NS(cached_tokens=400)),
              _hidden_params={})
    kw = {"model": "gpt-4o", "litellm_params": {"custom_llm_provider": "openai"}, "litellm_call_id": "abc"}
    lg.log_success_event(kw, resp, None, None)
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=600, cached_input_tokens=400, output_tokens=200)
    assert lg.totals()["kg_co2e"] == d.kg_co2e
    assert resp._hidden_params["saef"]["kg_co2e"] == d.kg_co2e
    assert json.loads(log.read_text())["call_id"] == "abc"

    bedrock = {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
               "litellm_params": {"custom_llm_provider": "bedrock", "aws_region_name": "eu-north-1"}}
    r2 = NS(usage=NS(prompt_tokens=100, completion_tokens=10), _hidden_params={})
    lg.log_success_event(bedrock, r2, None, None)
    assert r2._hidden_params["saef"]["region_id"] == "eu-north-1"

    azure = {"model": "gpt-4o", "litellm_params": {"custom_llm_provider": "azure"}}
    r3 = NS(usage=NS(prompt_tokens=100, completion_tokens=10), _hidden_params={})
    lg.log_success_event(azure, r3, None, None)
    assert r3._hidden_params["saef"]["region_id"] == "swedencentral"
    assert lg.totals()["calls"] == 3


def test_litellm_async_and_fallback_to_standard_payload():
    pytest.importorskip("litellm")
    import asyncio

    from saef.integrations.litellm import SaefLogger

    lg = SaefLogger()
    kw = {"model": "gpt-4o", "standard_logging_object": {"prompt_tokens": 50, "completion_tokens": 5}}
    asyncio.run(lg.async_log_success_event(kw, NS(), None, None))
    assert lg.totals()["calls"] == 1
    lg.log_success_event({"model": "gpt-4o"}, NS(), None, None)
    assert lg.totals()["calls"] == 1


def test_litellm_real_mock_completion():
    litellm = pytest.importorskip("litellm")
    import time

    from saef.integrations.litellm import SaefLogger

    lg = SaefLogger(region="eu-west-1")
    old = litellm.callbacks
    litellm.callbacks = [lg]
    try:
        r = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="hello")
        for _ in range(50):
            if lg.totals()["calls"]:
                break
            time.sleep(0.1)
    finally:
        litellm.callbacks = old
    assert lg.totals()["calls"] == 1
    assert r._hidden_params["saef"]["region_id"] == "eu-west-1"
    assert lg.totals()["cost"] > 0 and lg.totals()["g_co2e_per_usd"] > 0


def _langchain_result(**kw):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    msg = AIMessage(content="hi", **kw.pop("msg", {}))
    return LLMResult(generations=[[ChatGeneration(message=msg)]], **kw)


def test_langchain_usage_metadata_with_cache():
    pytest.importorskip("langchain_core")
    from saef.integrations.langchain import SaefCallbackHandler

    h = SaefCallbackHandler(region="eu-west-1")
    rid = uuid.uuid4()
    h.on_chat_model_start({}, [[]], run_id=rid, invocation_params={"model": "gpt-4o"})
    res = _langchain_result(msg={"usage_metadata": {
        "input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200, "input_token_details": {"cache_read": 400}}})
    h.on_llm_end(res, run_id=rid)
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=600, cached_input_tokens=400, output_tokens=200)
    assert h.totals()["kg_co2e"] == d.kg_co2e
    assert h.totals()["calls"] == 1


def test_langchain_llm_output_token_usage_and_model_name(tmp_path):
    pytest.importorskip("langchain_core")
    from langchain_core.outputs import Generation, LLMResult

    from saef.integrations.langchain import SaefCallbackHandler

    log = tmp_path / "lc.jsonl"
    h = SaefCallbackHandler(region="eu-west-1", log_path=str(log))
    res = LLMResult(generations=[[Generation(text="x")]],
                    llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}, "model_name": "gpt-4o-mini"})
    h.on_llm_end(res, run_id=uuid.uuid4())
    d = direct(model="gpt-4o-mini", region="eu-west-1", input_tokens=100, output_tokens=20)
    assert h.totals()["kg_co2e"] == d.kg_co2e
    assert len(log.read_text().splitlines()) == 1


def test_langchain_skips_without_model_or_usage():
    pytest.importorskip("langchain_core")
    from langchain_core.outputs import Generation, LLMResult

    from saef.integrations.langchain import SaefCallbackHandler

    h = SaefCallbackHandler()
    h.on_llm_end(LLMResult(generations=[[Generation(text="x")]], llm_output={"token_usage": {"prompt_tokens": 1}}), run_id=uuid.uuid4())
    h.on_llm_end(LLMResult(generations=[[Generation(text="x")]], llm_output={"model_name": "gpt-4o"}), run_id=uuid.uuid4())
    assert h.totals()["calls"] == 0


def _fake_agent_parts(replies):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    msgs = [AIMessage(content="Hi there", response_metadata={"model_name": "gpt-4o"}, usage_metadata=u) for u in replies]
    return create_agent, GenericFakeChatModel(messages=iter(msgs))


_USAGE = {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500, "input_token_details": {"cache_read": 800}}


def test_langchain_middleware_agent_run_attaches_and_totals(tmp_path):
    pytest.importorskip("langchain.agents")
    from saef.integrations.langchain import SaefMiddleware

    create_agent, llm = _fake_agent_parts([_USAGE])
    log = tmp_path / "mw.jsonl"
    mw = SaefMiddleware(region="eu-west-1", attach=True, log_path=str(log))
    out = create_agent(llm, tools=[], middleware=[mw]).invoke({"messages": [("user", "Hello")]})
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=400, cached_input_tokens=800, output_tokens=300)
    t = mw.totals()
    assert t["calls"] == 1 and t["kg_co2e"] == d.kg_co2e
    assert out["messages"][-1].response_metadata["saef"]["kg_co2e"] == d.kg_co2e
    assert len(log.read_text().splitlines()) == 1


def test_langchain_middleware_no_attach_by_default_and_async():
    pytest.importorskip("langchain.agents")
    import asyncio

    from saef.integrations.langchain import SaefMiddleware

    create_agent, llm = _fake_agent_parts([_USAGE, _USAGE])
    mw = SaefMiddleware(region="eu-west-1")
    agent = create_agent(llm, tools=[], middleware=[mw])
    out = agent.invoke({"messages": [("user", "Hello")]})
    assert "saef" not in out["messages"][-1].response_metadata
    asyncio.run(agent.ainvoke({"messages": [("user", "Again")]}))
    assert mw.totals()["calls"] == 2
    mw.reset()
    assert mw.totals()["calls"] == 0


def test_langchain_middleware_skips_without_usage_and_falls_back_to_request_model():
    pytest.importorskip("langchain.agents")
    from langchain_core.messages import AIMessage

    from saef.integrations.langchain import SaefMiddleware

    class M:
        model_name = "gpt-4o"

    class R:
        model = M()

    mw = SaefMiddleware(region="eu-west-1")
    mw.wrap_model_call(R(), lambda r: AIMessage(content="x"))
    assert mw.totals()["calls"] == 0
    mw.wrap_model_call(R(), lambda r: AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}))
    assert mw.totals()["calls"] == 1


def _li_chat_end(raw=None, additional_kwargs=None, span_id="s1"):
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
    from llama_index.core.instrumentation.events.llm import LLMChatEndEvent

    msg = ChatMessage(role=MessageRole.ASSISTANT, content="hi")
    resp = ChatResponse(message=msg, raw=raw, additional_kwargs=additional_kwargs or {})
    return LLMChatEndEvent(messages=[msg], response=resp, span_id=span_id)


def test_llamaindex_chat_end_raw_usage_with_cache():
    pytest.importorskip("llama_index.core")
    from saef.integrations.llamaindex import SaefEventHandler

    h = SaefEventHandler(region="eu-west-1")
    raw = {"model": "gpt-4o", "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 400}}}
    h.handle(_li_chat_end(raw=raw))
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=600, cached_input_tokens=400, output_tokens=200)
    t = h.totals()
    assert t["calls"] == 1 and t["kg_co2e"] == d.kg_co2e and t["kg_co2e_high_case"] == d.kg_co2e_high_case


def test_llamaindex_model_from_start_event_and_additional_kwargs(tmp_path):
    pytest.importorskip("llama_index.core")
    from llama_index.core.instrumentation.events.llm import LLMCompletionEndEvent, LLMCompletionStartEvent
    from llama_index.core.base.llms.types import CompletionResponse

    from saef.integrations.llamaindex import SaefEventHandler

    log = tmp_path / "li.jsonl"
    h = SaefEventHandler(region="eu-west-1", log_path=str(log))
    h.handle(LLMCompletionStartEvent(prompt="p", additional_kwargs={}, model_dict={"model": "gpt-4o-mini"}, span_id="c1"))
    resp = CompletionResponse(text="x", additional_kwargs={"prompt_tokens": 100, "completion_tokens": 20})
    h.handle(LLMCompletionEndEvent(prompt="p", response=resp, span_id="c1"))
    d = direct(model="gpt-4o-mini", region="eu-west-1", input_tokens=100, output_tokens=20)
    assert h.totals()["kg_co2e"] == d.kg_co2e
    assert len(log.read_text().splitlines()) == 1
    assert h._models == {}


def test_llamaindex_skips_without_model_or_usage():
    pytest.importorskip("llama_index.core")
    from saef.integrations.llamaindex import SaefEventHandler

    h = SaefEventHandler(region="eu-west-1")
    h.handle(_li_chat_end(raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
    h.handle(_li_chat_end(raw={"model": "gpt-4o"}))
    h.handle(_li_chat_end(raw=None))
    assert h.totals()["calls"] == 0
    h.handle(_li_chat_end(raw={"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
    assert h.totals()["calls"] == 1
    h.reset()
    assert h.totals()["calls"] == 0


def test_llamaindex_dispatcher_end_to_end():
    pytest.importorskip("llama_index.core")
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata
    from llama_index.core.instrumentation import get_dispatcher
    from llama_index.core.llms import CustomLLM
    from llama_index.core.llms.callbacks import llm_chat_callback

    from saef.integrations.llamaindex import SaefEventHandler

    class FakeLLM(CustomLLM):
        @property
        def metadata(self):
            return LLMMetadata(model_name="gpt-4o")

        @llm_chat_callback()
        def chat(self, messages, **kwargs):
            raw = {"model": "gpt-4o", "usage": {"prompt_tokens": 50, "completion_tokens": 10}}
            return ChatResponse(message=ChatMessage(role="assistant", content="hi"), raw=raw)

        def complete(self, prompt, formatted=False, **kwargs):
            raise NotImplementedError

        def stream_complete(self, prompt, formatted=False, **kwargs):
            raise NotImplementedError

    h = SaefEventHandler(region="eu-west-1").register()
    try:
        FakeLLM().chat([ChatMessage(role="user", content="Hello")])
    finally:
        get_dispatcher().event_handlers.remove(h)
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=50, output_tokens=10)
    assert h.totals()["calls"] == 1 and h.totals()["kg_co2e"] == d.kg_co2e


def test_haystack_chat_replies_openai_and_anthropic_meta():
    pytest.importorskip("haystack")
    from haystack.dataclasses import ChatMessage

    from saef.integrations.haystack import SaefTracker

    t = SaefTracker(region="eu-west-1")
    openai_reply = ChatMessage.from_assistant("hi", meta={
        "model": "gpt-4o", "finish_reason": "stop",
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 400}}})
    anthropic_reply = ChatMessage.from_assistant("hi", meta={
        "model": "claude-3-5-sonnet-20241022", "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50}})
    out = t.run(replies=[openai_reply, anthropic_reply])
    assert out["replies"] == [openai_reply, anthropic_reply] and len(out["saef"]) == 2
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=600, cached_input_tokens=400, output_tokens=200)
    assert out["saef"][0]["kg_co2e"] == d.kg_co2e
    tot = t.totals()
    assert tot["calls"] == 2 and tot["cached_input_tokens"] == 450


def test_haystack_text_generator_meta_and_skips(tmp_path):
    pytest.importorskip("haystack")
    from haystack.dataclasses import ChatMessage

    from saef.integrations.haystack import SaefTracker

    log = tmp_path / "hs.jsonl"
    t = SaefTracker(region="eu-west-1", log_path=str(log))
    out = t.run(replies=["a", "b", "c"], meta=[
        {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        {"model": "gpt-4o-mini", "usage": {}},
    ])
    assert out["replies"] == ["a", "b", "c"] and len(out["saef"]) == 1
    d = direct(model="gpt-4o-mini", region="eu-west-1", input_tokens=100, output_tokens=20)
    assert t.totals()["kg_co2e"] == d.kg_co2e
    assert len(log.read_text().splitlines()) == 1
    t.run(replies=[ChatMessage.from_assistant("no meta")])
    t.run(replies=["x"])
    assert t.totals()["calls"] == 1
    t.reset()
    assert t.totals()["calls"] == 0


def test_haystack_in_pipeline():
    pytest.importorskip("haystack")
    from typing import List

    from haystack import Pipeline, component
    from haystack.dataclasses import ChatMessage

    from saef.integrations.haystack import SaefTracker

    @component
    class FakeChatGenerator:
        @component.output_types(replies=List[ChatMessage])
        def run(self, prompt: str):
            return {"replies": [ChatMessage.from_assistant("hi", meta={
                "model": "gpt-4o", "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}})]}

    pipe = Pipeline()
    pipe.add_component("llm", FakeChatGenerator())
    pipe.add_component("saef", SaefTracker(region="eu-west-1"))
    pipe.connect("llm.replies", "saef.replies")
    res = pipe.run({"llm": {"prompt": "Hello"}})
    d = direct(model="gpt-4o", region="eu-west-1", input_tokens=50, output_tokens=10)
    assert res["saef"]["saef"][0]["kg_co2e"] == d.kg_co2e
    assert pipe.get_component("saef").totals()["calls"] == 1
