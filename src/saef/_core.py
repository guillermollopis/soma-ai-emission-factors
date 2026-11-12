import csv
import fnmatch
import io
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Optional

DATASET_VERSION = "2.0.0"
DEFAULT_SCENARIO = "H100-central"
LIFECYCLE_TERMS = ("embodied_hardware", "training_amortised")

CLASS_ALIASES = {
    "A": "class_a_small",
    "B": "class_b_mid",
    "C": "class_c_large_moe",
}

BIBTEX = """@misc{llopis2026somafactors,
  author       = {Llopis-Garcia, Guillermo},
  title        = {{SOMA AI Emission Factors (SAEF)}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20443585},
  url          = {https://doi.org/10.5281/zenodo.20443585}
}

@misc{llopis2026accounting,
  author        = {Llopis-Garcia, Guillermo},
  title         = {Accounting for Purchased Artificial Intelligence Inference in Corporate Greenhouse Gas Inventories: A Token-Based Method},
  year          = {2026},
  eprint        = {2606.10660},
  archivePrefix = {arXiv},
  note          = {Accepted for publication in Sustainable Production and Consumption (Elsevier)}
}
"""


class SaefError(ValueError):
    pass


@lru_cache(maxsize=None)
def _read(name):
    text = resources.files("saef").joinpath("data", name).read_text(encoding="utf-8")
    return tuple(csv.DictReader(io.StringIO(text, newline="")))


def _region_key(region_id):
    return region_id.split(" (")[0].strip()


def regions():
    """Region ids available in the dataset, sorted."""
    return sorted({_region_key(r["region_id"]) for r in _read("carbon_factors.csv")})


def _region_labels():
    labels = {}
    for r in _read("carbon_factors.csv"):
        key = _region_key(r["region_id"])
        labels[key] = r["region_id"][len(key):].strip(" ()")
    return labels


def _resolve_region(region):
    available = regions()
    wanted = region.strip().lower()
    for rid in available:
        if rid.lower() == wanted:
            return rid
    by_label = [rid for rid, label in _region_labels().items() if label and wanted in label.lower()]
    if len(by_label) == 1:
        return by_label[0]
    prefixed = [rid for rid in available if rid.lower().startswith(wanted)]
    if len(prefixed) == 1:
        return prefixed[0]
    candidates = sorted(set(by_label) | set(prefixed))
    if candidates:
        raise SaefError(f"Region {region!r} is ambiguous, candidates: {', '.join(candidates)}")
    raise SaefError(f"Unknown region {region!r}. Available regions: {', '.join(available)}")


def _resolve_class(model_class):
    raw = str(model_class).strip()
    cls = CLASS_ALIASES.get(raw.upper(), raw)
    known = {r["model_class"] for r in _read("carbon_factors.csv")}
    if cls not in known:
        raise SaefError(
            f"No carbon factors for model class {model_class!r}. Available: {', '.join(sorted(known))} (or A, B, C)"
        )
    return cls


def _lookup_model(model):
    name = model.strip().lower().rsplit("/", 1)[-1]
    best, best_score = None, -1
    for r in _read("model_class_mapping.csv"):
        for pattern in r["model_pattern"].split(";"):
            pattern = pattern.strip().lower()
            if fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(name, pattern + "*"):
                score = len(pattern.replace("*", ""))
                if score > best_score:
                    best, best_score = r, score
    if best is None:
        raise SaefError(f"Model {model!r} is not in model_class_mapping.csv. Pass model_class=A|B|C explicitly.")
    return best


def model_class_for(model):
    """Map a model name such as 'gpt-4o' to its SAEF model class via model_class_mapping.csv."""
    return _lookup_model(model)["model_class"]


def _find(name, cls, scenario, rid):
    for r in _read(name):
        if r["model_class"] == cls and r["infra_scenario"] == scenario and _region_key(r["region_id"]) == rid:
            return r
    return None


def factor(model_class="B", region="us-east", scenario=DEFAULT_SCENARIO):
    """Total factor in kg CO2e per million output tokens, with its three terms, water and sources."""
    cls = _resolve_class(model_class)
    rid = _resolve_region(region)
    row = _find("carbon_factors.csv", cls, scenario, rid)
    if row is None:
        scenarios = sorted(
            {
                r["infra_scenario"]
                for r in _read("carbon_factors.csv")
                if r["model_class"] == cls and _region_key(r["region_id"]) == rid
            }
        )
        raise SaefError(
            f"No {scenario!r} row for {cls} in {rid}. Scenarios available for this class and region: {', '.join(scenarios)}"
        )
    terms = {"electricity": float(row["kg_co2e_per_1m_tokens"])}
    sources = {
        "electricity": {
            "file": "carbon_factors.csv",
            "column": "kg_co2e_per_1m_tokens",
            "row": f"{row['model_class']} | {row['infra_scenario']} | {row['region_id']}",
        }
    }
    for term in LIFECYCLE_TERMS:
        lc = next((r for r in _read("lifecycle_factors.csv") if r["model_class"] == cls and r["term"] == term), None)
        if lc is None:
            raise SaefError(f"lifecycle_factors.csv has no {term} row for {cls}")
        terms[term] = float(lc["kg_co2e_per_1m_tokens"])
        sources[term] = {
            "file": "lifecycle_factors.csv",
            "column": "kg_co2e_per_1m_tokens",
            "row": f"{cls} | {term}",
        }
    water = None
    wrow = _find("water_factors.csv", cls, scenario, rid)
    if wrow is not None:
        water = float(wrow["total_ml_per_1k_tokens"]) * 1000
        sources["water"] = {
            "file": "water_factors.csv",
            "column": "total_ml_per_1k_tokens",
            "row": f"{wrow['model_class']} | {wrow['infra_scenario']} | {wrow['region_id']}",
        }
    return {
        "model_class": cls,
        "region": rid,
        "scenario": scenario,
        "kg_co2e_per_1m_tokens": sum(terms.values()),
        "terms": terms,
        "water_ml_per_1m_tokens": water,
        "version": DATASET_VERSION,
        "sources": sources,
    }


@dataclass(frozen=True)
class Estimate:
    kg_co2e: float
    water_ml: Optional[float]
    tokens: float
    effective_tokens: float
    reasoning_multiplier: float
    model: Optional[str]
    model_class: str
    region: str
    scenario: str
    terms_kg_co2e: dict
    factor: dict = field(repr=False)
    sources: dict = field(repr=False)
    version: str = DATASET_VERSION
    note: str = "Estimate, not a measurement."

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def estimate(
    tokens,
    model=None,
    model_class=None,
    region="global-average",
    scenario=DEFAULT_SCENARIO,
    reasoning_multiplier=None,
):
    """Estimate kg CO2e (and water in mL when available) for a number of output tokens.

    Give model or model_class. When model is given, its reasoning multiplier from
    model_class_mapping.csv scales the tokens (hidden reasoning tokens); override it with reasoning_multiplier.
    """
    if tokens < 0:
        raise SaefError("tokens must be >= 0")
    if (model is None) == (model_class is None):
        raise SaefError("Pass exactly one of model or model_class")
    extra = {}
    multiplier = 1.0
    if model is not None:
        mrow = _lookup_model(model)
        model_class = mrow["model_class"]
        multiplier = float(mrow["reasoning_multiplier"])
        extra["model_class"] = {
            "file": "model_class_mapping.csv",
            "column": "model_class, reasoning_multiplier",
            "row": mrow["model_pattern"],
        }
    if reasoning_multiplier is not None:
        multiplier = float(reasoning_multiplier)
    f = factor(model_class, region, scenario)
    effective = tokens * multiplier
    millions = effective / 1e6
    terms = {k: v * millions for k, v in f["terms"].items()}
    water = f["water_ml_per_1m_tokens"]
    return Estimate(
        kg_co2e=sum(terms.values()),
        water_ml=None if water is None else water * millions,
        tokens=tokens,
        effective_tokens=effective,
        reasoning_multiplier=multiplier,
        model=model,
        model_class=f["model_class"],
        region=f["region"],
        scenario=f["scenario"],
        terms_kg_co2e=terms,
        factor=f,
        sources={**f["sources"], **extra},
    )


def citation():
    """BibTeX for the dataset (Zenodo concept DOI 10.5281/zenodo.20443585) and the method paper."""
    return BIBTEX
