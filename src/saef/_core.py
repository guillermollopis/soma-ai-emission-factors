from dataclasses import dataclass, field
from typing import Optional

from . import _calc
from ._calc import SaefError

DATASET_VERSION = "3.0.0"

BIBTEX = """@misc{llopis2026somafactors,
  author       = {Llopis-Garcia, Guillermo},
  title        = {{SOMA AI Emission Factors (SAEF)}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3.0.0},
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


def regions(provider=None):
    """Cloud region ids (AWS, Azure, Google Cloud), sorted. Pass provider='aws'|'azure'|'gcp' to filter."""
    rows = _calc.read("cloud_region_factors.csv")
    if provider:
        rows = [r for r in rows if r["provider"].lower() == provider.strip().lower()]
    return sorted({r["region_id"] for r in rows})


def grids():
    """Grid ids usable as a region: ISO-3 country codes, eGRID subregions, EU27, WORLD and one subnational row."""
    return sorted(r["grid_id"] for r in _calc.read("grid_intensity_all.csv"))


def model_class_for(model):
    """Map a model identifier such as 'gpt-4o' to its class (A, B or C) via model_class_mapping.csv.

    An identifier that matches no pattern is assigned the conservative class C, as the dataset specifies."""
    m = _calc.map_model(model)
    if m["out_of_scope"]:
        raise SaefError(f"{model!r} is out of scope (embeddings, image, audio, video or moderation)")
    return m["model_class"]


def factor(model_class="B", region="us-east-1", provider=None):
    """Class factor at one region, H100 central case.

    kg_co2e_per_1m_billed_tokens is the headline for a billed total with the default input share.
    terms are kg CO2e per million EFFECTIVE tokens (electricity, embodied, construction), plus training per million billed tokens."""
    cl = _calc.emissions(model_class=model_class, provider=provider, region=region, billed_tokens=1e6)
    f = cl["factors"]
    return {
        "model_class": cl["model_class"],
        "region_id": cl["region_id"],
        "grid_id": cl["grid_id"],
        "grid_kg_co2e_per_kwh": cl["grid_kg_co2e_per_kwh"],
        "kg_co2e_per_1m_billed_tokens": f["per_billed"],
        "kg_co2e_per_1m_effective_tokens": f["footprint"],
        "kg_co2e_per_1m_effective_tokens_high_case": f["footprint_high"],
        "terms": {
            "electricity": f["electricity"],
            "embodied": f["embodied"],
            "construction": f["construction"],
            "training_per_1m_billed_tokens": f["training"],
        },
        "effective_per_billed_tokens": _calc.effective_from_billed(1.0),
        "version": DATASET_VERSION,
        "warnings": cl["warnings"],
        "provenance": cl["provenance"],
    }


@dataclass(frozen=True)
class Estimate:
    kg_co2e: float
    kg_co2e_high_case: float
    development_line_kg_co2e: float
    terms_kg_co2e: dict
    water_ml: Optional[float]
    water_ml_excl_hydro: Optional[float]
    model: Optional[str]
    model_class: str
    class_basis: str
    region_id: str
    grid_kg_co2e_per_kwh: float
    billed_tokens: float
    effective_tokens: float
    warnings: list = field(repr=False)
    provenance: list = field(repr=False)
    water: dict = field(repr=False)
    version: str = DATASET_VERSION
    note: str = "Estimate, not a measurement. The high case is a conservative bound; the development line is reported separately and is not part of kg_co2e."

    @property
    def sources(self):
        return self.provenance

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def estimate(
    tokens=None,
    model=None,
    model_class=None,
    region=None,
    provider=None,
    output_tokens=None,
    input_tokens=0.0,
    cached_input_tokens=0.0,
):
    """Estimate kg CO2e (and water in mL where the region has a water factor) for AI inference.

    Pass tokens, the BILLED total (input plus output), or pass output_tokens with input_tokens and
    cached_input_tokens for a billing-export split. Pass model or model_class. region may be a cloud
    region id, an ISO-3 country code, an eGRID subregion, 'eu-27' or None for the global average.
    Every number is traced in provenance by file, line and column.
    """
    if (tokens is None) == (output_tokens is None):
        raise SaefError("Pass either tokens (billed total) or output_tokens (with input_tokens)")
    given = tokens if tokens is not None else output_tokens
    if given < 0 or input_tokens < 0 or cached_input_tokens < 0:
        raise SaefError("token counts must be >= 0")
    r = _calc.emissions(
        model=model, model_class=model_class, provider=provider, region=region,
        billed_tokens=tokens, output_tokens=output_tokens or 0.0,
        input_tokens=input_tokens, cached_input_tokens=cached_input_tokens,
    )
    w = _calc.water(r["effective_tokens"], r["model_class"], provider, region)
    return Estimate(
        kg_co2e=r["kg_co2e"],
        kg_co2e_high_case=r["kg_co2e_high_case"],
        development_line_kg_co2e=r["development_line_kg_co2e"],
        terms_kg_co2e={
            "electricity": r["electricity_kg_co2e"],
            "embodied": r["embodied_kg_co2e"],
            "construction": r["construction_kg_co2e"],
            "training": r["training_kg_co2e"],
        },
        water_ml=w["ml_total_reig"] if w["available"] else None,
        water_ml_excl_hydro=w["ml_total_excl_hydro"] if w["available"] else None,
        model=model,
        model_class=r["model_class"],
        class_basis=r["class_basis"],
        region_id=r["region_id"],
        grid_kg_co2e_per_kwh=r["grid_kg_co2e_per_kwh"],
        billed_tokens=r["billed_tokens"],
        effective_tokens=r["effective_tokens"],
        warnings=list(dict.fromkeys(r["warnings"] + w["warnings"])),
        provenance=r["provenance"] + w["provenance"],
        water=w,
    )


def citation():
    """BibTeX for the dataset (Zenodo concept DOI 10.5281/zenodo.20443585) and the method paper."""
    return BIBTEX
