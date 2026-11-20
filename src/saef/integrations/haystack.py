"""Haystack 2.x component that estimates carbon and water for the replies of the generator before it.

    from haystack import Pipeline
    from haystack.components.generators.chat import OpenAIChatGenerator
    from saef.integrations.haystack import SaefTracker

    pipe = Pipeline()
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
    pipe.add_component("saef", SaefTracker(region="eu-west-1"))
    pipe.connect("llm.replies", "saef.replies")
    result = pipe.run({"llm": {"messages": [...]}})
    result["saef"]["saef"]
    pipe.get_component("saef").totals()

For text generators such as OpenAIGenerator, also connect "llm.meta" to "saef.meta".
"""
from typing import Any, Dict, List, Optional, Union

from haystack import component
from haystack.dataclasses import ChatMessage

from .. import SaefError
from ._base import Tracker
from .openai import usage_from_response


@component
class SaefTracker:
    """Pass-through component. Takes the replies of a ChatGenerator, or the replies and meta of a text Generator,
    and outputs the replies unchanged plus one SAEF estimate dict per reply that has a model name and token usage.

    region defaults to env SAEF_REGION, then 'global-average'. Usage is read from meta["usage"], either OpenAI style
    (prompt_tokens, completion_tokens) or Anthropic style (input_tokens, output_tokens), and the model from
    meta["model"]. Replies without a model name or without token usage are not counted."""

    def __init__(self, region: Optional[str] = None, log_path: Optional[str] = None):
        self._tracker = Tracker(region=region, log_path=log_path)

    def totals(self) -> Dict[str, Any]:
        return self._tracker.totals()

    def reset(self) -> None:
        self._tracker.reset()

    @component.output_types(replies=List[Union[ChatMessage, str]], saef=List[Dict[str, Any]])
    def run(self, replies: List[Union[ChatMessage, str]], meta: Optional[List[Dict[str, Any]]] = None):
        records = []
        for i, reply in enumerate(replies):
            m = reply.meta if isinstance(reply, ChatMessage) else (meta[i] if meta and i < len(meta) else None)
            rec = self._record(m or {})
            if rec is not None:
                records.append(rec)
        return {"replies": replies, "saef": records}

    def _record(self, meta):
        model = meta.get("model")
        if not model or not meta.get("usage"):
            return None
        try:
            _, inp, cached, out = usage_from_response({"model": model, "usage": meta["usage"]})
        except SaefError:
            return None
        if inp + cached + out <= 0:
            return None
        return self._tracker.record(model, inp, out, cached)
