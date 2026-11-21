import json
import os
import threading

from .. import SaefError, estimate

DEFAULT_REGION = "global-average"


def valid_cost(cost):
    return isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost == cost and cost >= 0


def default_region():
    return os.environ.get("SAEF_REGION") or DEFAULT_REGION


class Tracker:
    """Thread-safe running totals of per-call estimates, with optional JSON lines output."""

    def __init__(self, region=None, provider=None, log_path=None):
        self.region = region or default_region()
        self.provider = provider
        self.log_path = log_path
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._calls = 0
            self._skipped = 0
            self._input = self._cached = self._output = 0
            self._kg = self._kg_high = self._water = 0.0
            self._water_missing = 0
            self._cost = {}
            self._cost_calls = 0
            self._kg_costed = 0.0
            self._by_model = {}

    def _estimate(self, model, input_tokens, output_tokens, cached_input_tokens, region, provider):
        kw = dict(
            model=model, provider=provider or self.provider, input_tokens=input_tokens,
            output_tokens=output_tokens, cached_input_tokens=cached_input_tokens,
        )
        try:
            return estimate(region=region or self.region, **kw)
        except SaefError:
            if region and region != self.region:
                return estimate(region=self.region, **kw)
            raise

    def record(self, model, input_tokens, output_tokens, cached_input_tokens=0, region=None, provider=None, extra=None,
               cost=None, cost_currency="USD"):
        """Estimate one call, add it to the totals and return a plain dict, or None if the model is out of scope.

        cost is the price of the call as reported by the framework, never computed here. It is stored next to the
        estimate only when it is a number >= 0.

        A per-call region that SAEF does not know falls back to the configured region."""
        try:
            e = self._estimate(model, input_tokens, output_tokens, cached_input_tokens, region, provider)
        except SaefError:
            with self._lock:
                self._skipped += 1
            return None
        out = {
            "model": model,
            "model_class": e.model_class,
            "region_id": e.region_id,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "kg_co2e": e.kg_co2e,
            "kg_co2e_high_case": e.kg_co2e_high_case,
            "water_ml": e.water_ml,
            "dataset_version": e.version,
            "note": "estimate",
        }
        has_cost = valid_cost(cost)
        if has_cost:
            out["cost"] = float(cost)
            out["cost_currency"] = cost_currency
        with self._lock:
            self._calls += 1
            self._input += input_tokens
            self._cached += cached_input_tokens
            self._output += output_tokens
            self._kg += e.kg_co2e
            self._kg_high += e.kg_co2e_high_case
            if e.water_ml is None:
                self._water_missing += 1
            else:
                self._water += e.water_ml
            if has_cost:
                self._cost[cost_currency] = self._cost.get(cost_currency, 0.0) + cost
                self._cost_calls += 1
                self._kg_costed += e.kg_co2e
            m = self._by_model.setdefault(model, {"calls": 0, "kg_co2e": 0.0})
            m["calls"] += 1
            m["kg_co2e"] += e.kg_co2e
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(out, **(extra or {}))) + "\n")
        return out

    def totals(self):
        with self._lock:
            totals = {
                "calls": self._calls,
                "skipped_calls": self._skipped,
                "input_tokens": self._input,
                "cached_input_tokens": self._cached,
                "output_tokens": self._output,
                "kg_co2e": self._kg,
                "kg_co2e_high_case": self._kg_high,
                "water_ml": self._water,
                "calls_without_water_factor": self._water_missing,
                "by_model": {k: dict(v) for k, v in self._by_model.items()},
            }
            if self._cost_calls:
                totals["calls_with_cost"] = self._cost_calls
                if len(self._cost) == 1:
                    (currency, total), = self._cost.items()
                    totals["cost"] = total
                    totals["cost_currency"] = currency
                    totals["g_co2e_per_usd"] = self._kg_costed * 1000 / total if currency == "USD" and total > 0 else None
                else:
                    totals["cost_by_currency"] = dict(self._cost)
            return totals
