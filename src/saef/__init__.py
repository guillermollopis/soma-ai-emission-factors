"""SOMA AI Emission Factors (SAEF): estimated carbon and water factors for AI inference."""
from ._core import (
    DATASET_VERSION,
    Estimate,
    SaefError,
    citation,
    estimate,
    factor,
    grids,
    model_class_for,
    regions,
)

__version__ = "3.6.0"


def _get(rec, key):
    return rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def summarize(records):
    """Totals over per-call records: Estimate objects or the dicts the integrations produce and log.

    Returns kg_co2e, water_ml, calls, plus kwh when any record carries kwh. When records carry cost (reported by
    your framework, never priced here) it also returns cost, cost_currency, calls_with_cost and g_co2e_per_usd, the
    central estimate in grams over the cost, computed on the calls that have a cost. g_co2e_per_usd is None when the
    total cost is zero or the currency is not USD. Mixed currencies give cost_by_currency instead of cost."""
    out = {"calls": 0, "kg_co2e": 0.0, "water_ml": 0.0}
    kwh = None
    costs = {}
    cost_calls = 0
    kg_costed = 0.0
    for r in records:
        out["calls"] += 1
        kg = _get(r, "kg_co2e")
        kg = kg if _is_num(kg) else 0.0
        out["kg_co2e"] += kg
        w = _get(r, "water_ml")
        if _is_num(w):
            out["water_ml"] += w
        k = _get(r, "kwh")
        if _is_num(k):
            kwh = (kwh or 0.0) + k
        c = _get(r, "cost")
        if _is_num(c) and c >= 0:
            cur = _get(r, "cost_currency") or "USD"
            costs[cur] = costs.get(cur, 0.0) + c
            cost_calls += 1
            kg_costed += kg
    if kwh is not None:
        out["kwh"] = kwh
    if cost_calls:
        out["calls_with_cost"] = cost_calls
        if len(costs) == 1:
            (cur, total), = costs.items()
            out["cost"] = total
            out["cost_currency"] = cur
            out["g_co2e_per_usd"] = kg_costed * 1000 / total if cur == "USD" and total > 0 else None
        else:
            out["cost_by_currency"] = costs
    return out

__all__ = [
    "DATASET_VERSION",
    "Estimate",
    "SaefError",
    "__version__",
    "citation",
    "estimate",
    "factor",
    "grids",
    "model_class_for",
    "regions",
    "summarize",
]
