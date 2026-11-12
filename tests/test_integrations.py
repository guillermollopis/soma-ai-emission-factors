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
