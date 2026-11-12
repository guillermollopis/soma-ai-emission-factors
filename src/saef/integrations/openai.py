"""Estimates from API response objects. Duck-typed, so no SDK import is needed.

Handles OpenAI Chat Completions, OpenAI Responses and Anthropic Messages objects, or the same shapes as dicts."""
from .. import SaefError, estimate
from ._base import default_region


def _get(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _n(obj, name):
    v = _get(obj, name)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def usage_from_response(resp):
    """Return (model, uncached_input_tokens, cached_input_tokens, output_tokens) from a response object or dict.

    OpenAI counts cached tokens inside the prompt total. Anthropic reports cache reads and cache writes beside
    input_tokens. Cache writes are treated as ordinary uncached input and only cache reads get the cache credit."""
    usage = _get(resp, "usage")
    if usage is None:
        raise SaefError("response has no usage block")
    model = _get(resp, "model")
    if _get(usage, "prompt_tokens") is not None:
        prompt, out = _n(usage, "prompt_tokens"), _n(usage, "completion_tokens")
        cached = _n(_get(usage, "prompt_tokens_details"), "cached_tokens")
        return model, max(prompt - cached, 0), cached, out
    if _get(usage, "input_tokens") is None:
        raise SaefError("usage has neither prompt_tokens nor input_tokens")
    inp, out = _n(usage, "input_tokens"), _n(usage, "output_tokens")
    details = _get(usage, "input_tokens_details")
    if details is not None or _get(resp, "object") == "response":
        cached = _n(details, "cached_tokens")
        return model, max(inp - cached, 0), cached, out
    read, write = _n(usage, "cache_read_input_tokens"), _n(usage, "cache_creation_input_tokens")
    return model, inp + write, read, out


def estimate_from_response(resp, region=None, provider=None, model=None):
    """Estimate for one response object. The model comes from resp.model unless model= is passed."""
    m, inp, cached, out = usage_from_response(resp)
    model = model or m
    if not model:
        raise SaefError("response has no model, pass model=")
    return estimate(
        model=model, region=region or default_region(), provider=provider,
        input_tokens=inp, cached_input_tokens=cached, output_tokens=out,
    )
