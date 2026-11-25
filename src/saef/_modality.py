"""Estimates for non-text model types (embeddings, rerank, moderation, images, transcription, speech, voice, video) from modality_factors.csv.

Energy is stored per functional unit at GPU level and carried through the text chain (host, idle, PUE, grid, water).
Construction and embodied are charged per server kWh at the class A ratio. No training term is charged."""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from . import _calc
from ._calc import SaefError, prov

CASES = ("central", "low", "high")
# usage unit -> (functional unit of modality_factors.csv, functional units per usage unit)
UNITS = {
    "input_tokens": ("1k_input_tokens", 1e-3),
    "cached_input_tokens": ("1k_cached_input_tokens", 1e-3),
    "text_output_tokens": ("1k_text_output_tokens", 1e-3),
    "image_output_tokens": ("1k_image_output_tokens", 1e-3),
    "images": ("image", 1.0),
    "images_1024_low": ("image_1024_low", 1.0),
    "images_1024_medium": ("image_1024_medium", 1.0),
    "images_1024_high": ("image_1024_high", 1.0),
    "images_4k": ("image_4k", 1.0),
    "search_units": ("search_unit", 1.0),
    "calls": ("call", 1.0),
    "audio_minutes": ("audio_minute", 1.0),
    "audio_seconds": ("audio_minute", 1 / 60),
    "characters": ("1k_characters", 1e-3),
    "audio_output_tokens": ("1k_audio_output_tokens", 1e-3),
    "user_audio_minutes": ("user_audio_minute", 1.0),
    "assistant_audio_minutes": ("assistant_audio_minute", 1.0),
    "context_minutes": ("context_minute", 1.0),
    "conversation_minutes": ("conversation_minute", 1.0),
    "active_session_minutes": ("active_session_minute", 1.0),
    "video_seconds": ("video_second_720p", 1.0),
    "video_seconds_480p": ("video_second_480p", 1.0),
    "video_seconds_720p": ("video_second_720p", 1.0),
    "video_seconds_1080p": ("video_second_1080p", 1.0),
    "video_seconds_2160p": ("video_second_2160p", 1.0),
    "video_clips": ("video_clip", 1.0),
}
VIDEO_UNITS = tuple(u for u in UNITS if u.startswith("video_"))
VOICE_MINUTES = ("user_audio_minutes", "assistant_audio_minutes", "context_minutes", "conversation_minutes")
ALIASES = {
    "embedding_tokens": "input_tokens", "tokens_scored": "input_tokens",
    "audio_input_tokens": "input_tokens", "text_input_tokens": "input_tokens", "image_input_tokens": "input_tokens",
}
# Billing units that name a model type by themselves, with the class charged when the identifier does not fit the unit.
UNIT_DEFAULTS = {
    "image_output_tokens": ("image_generation_token_billed", "IMG-TOK-C"),
    "images": ("image_generation_diffusion", "D-large"),
    "images_1024_low": ("image_generation_token_billed", "IMG-TOK-C"),
    "images_1024_medium": ("image_generation_token_billed", "IMG-TOK-C"),
    "images_1024_high": ("image_generation_token_billed", "IMG-TOK-C"),
    "images_4k": ("image_generation_token_billed", "IMG-TOK-C"),
    "search_units": ("rerank", "R-large"),
    "calls": ("moderation", "M-B"),
    "audio_minutes": ("speech_to_text", "STT-LLM-B"),
    "audio_seconds": ("speech_to_text", "STT-LLM-B"),
    "characters": ("tts", "TTS-B"),
    "audio_output_tokens": ("realtime_turn_based", "RT-C"),
    "active_session_minutes": ("realtime_full_duplex", "FD-C"),
    **{u: ("realtime_turn_based", "RT-C") for u in VOICE_MINUTES},
    **{u: ("video_generation", "V-C") for u in VIDEO_UNITS},
}
INPUT_ONLY = {"embedding_text", "embedding_image", "rerank"}
OUTPUT_UNIT = {
    "image_generation_token_billed": "image_output_tokens",
    "speech_to_text": "text_output_tokens",
    "moderation": "text_output_tokens",
    "tts": "audio_output_tokens",
    "realtime_turn_based": "audio_output_tokens",
}
LOOSE_IMAGE_UNITS = ("images", "images_1024_low", "images_1024_medium", "images_1024_high", "images_4k")
MEGAPIXELS_1024 = 1.048576
NOTE = (
    "Estimate, not a measurement. low and high are the class range of the GPU energy at the central overhead; the high "
    "case is the central energy at the conservative overhead. No training term is charged for this model type."
)


@dataclass(frozen=True)
class ModalityEstimate:
    kg_co2e: float
    kg_co2e_low: float
    kg_co2e_high: float
    kg_co2e_high_case: float
    terms_kg_co2e: dict
    energy_wh: float
    energy_wh_low: float
    energy_wh_high: float
    gpu_energy_wh: float
    server_energy_wh: float
    water_ml: Optional[float]
    water_ml_low: Optional[float]
    water_ml_high: Optional[float]
    water_ml_excl_hydro: Optional[float]
    model: Optional[str]
    model_type: str
    model_class: str
    class_basis: str
    region_id: str
    grid_kg_co2e_per_kwh: float
    usage: dict
    components: list = field(repr=False)
    warnings: list = field(repr=False)
    provenance: list = field(repr=False)
    water: dict = field(repr=False)
    version: str
    billed_tokens: Optional[float] = None
    effective_tokens: Optional[float] = None
    note: str = NOTE

    @property
    def sources(self):
        return self.provenance

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@lru_cache(maxsize=None)
def _rows():
    return {(r["model_type"], r["class"], r["functional_unit"]): r for r in _calc.read("modality_factors.csv")}


@lru_cache(maxsize=None)
def _classes():
    out = {}
    for t, c, _u in _rows():
        out.setdefault(t, {})[c.lower()] = c
    return out


def canonical_unit(unit):
    u = str(unit).strip().lower().replace("-", "_").replace(" ", "_")
    u = ALIASES.get(u, u)
    for name in (u, u + "s"):
        if name in UNITS:
            return name, 1.0
    for name, (functional, scale) in UNITS.items():
        if u in (functional, "1k_" + name):
            return name, 1 / scale
    raise SaefError(f"unknown usage unit {unit!r}; use one of {', '.join(sorted(UNITS))}")


def _pick(types, cls, unit):
    return next((_rows()[t, cls, UNITS[unit][0]] for t in types if (t, cls, UNITS[unit][0]) in _rows()), None)


def _mapping(model, model_type, model_class):
    if model is not None:
        if model_type is not None or model_class is not None:
            raise SaefError("Pass model, or model_type with model_class, not both")
        return _calc.map_model(model)
    cls = _classes().get(model_type, {}).get(str(model_class).strip().lower())
    if cls is None:
        raise SaefError(f"no modality factors for model_type {model_type!r} class {model_class!r}")
    return dict(model_type=model_type, model_class=cls, basis="caller-supplied", out_of_scope=False,
                is_default_fallback=False, warnings=[], provenance=[])


def _resolve(model, mm, unit, warnings):
    """Factor row for one usage unit. A billing unit that names a model type decides before the identifier does."""
    if mm["out_of_scope"]:
        raise SaefError(f"{model!r} is out of scope: no model type covers it yet. {mm['warnings'][-1]}")
    unit_row = _calc.unit_only_row(model) if model is not None else None
    if unit_row is not None:
        row = _pick((unit_row["model_type"],), unit_row["model_class"], unit)
        if row is not None:
            warnings.append(f"{model!r} is billed in {unit}: routed by the billing unit to {row['model_type']}, class {row['class']}")
            return row
    types = mm["model_type"].split(";")
    if mm["model_type"] != "text":
        row = _pick(types, mm["model_class"], unit)
        if row is not None:
            return row
        supported = {t for t, _c, u in _rows() if u == UNITS[unit][0]}
        if unit not in UNIT_DEFAULTS or supported & set(types) or model is None:
            have = sorted(n for n in UNITS if _pick(types, mm["model_class"], n) is not None)
            raise SaefError(
                f"{model or mm['model_type']!r} (model type {mm['model_type']}, class {mm['model_class']}) is not charged "
                f"on {unit}; pass one of {', '.join(have)}"
            )
    elif unit not in UNIT_DEFAULTS:
        raise SaefError(f"{model!r} maps to the text factors; use estimate() for {unit}")
    t, cls = UNIT_DEFAULTS[unit]
    warnings.append(
        f"{model!r} does not map to a model type billed in {unit}: routed by the billing unit to {t}, "
        f"default class {cls}, and flagged"
    )
    return _pick((t,), cls, unit)


def estimate(model=None, usage=None, region=None, provider=None, megapixels=None, model_type=None, model_class=None,
             version=None):
    counts = {}
    for key, v in (usage or {}).items():
        if v is None:
            continue
        unit, scale = canonical_unit(key)
        v = float(v) * scale
        if not v >= 0:
            raise SaefError("usage counts must be >= 0")
        if v == float("inf"):
            raise SaefError("usage counts must be finite")
        counts[unit] = counts.get(unit, 0.0) + v
    if not counts:
        raise SaefError("usage is empty; pass at least one unit and its count")
    mm = _mapping(model, model_type, model_class)
    warnings, p = [], []
    reg = _calc.resolve_region(provider, region)
    warnings += reg["warnings"]
    p += mm["provenance"] + reg["provenance"]

    if "conversation_minutes" in counts and len(counts) > 1:
        raise SaefError("conversation_minutes already covers the whole call; do not combine it with other units")
    floor_images = None
    if "image_output_tokens" in counts:
        if any(u in counts for u in LOOSE_IMAGE_UNITS[1:]):
            raise SaefError("image_output_tokens and a per-image quality unit would count the same images twice; pass one")
        floor_images = counts.pop("images", None)

    components = []
    for unit, count in counts.items():
        row = _resolve(model, mm, unit, warnings)
        n = count * UNITS[unit][1]
        if unit == "image_output_tokens":
            components.append(_image_tokens(row, count, floor_images, warnings, p))
            continue
        if megapixels is not None and row["model_type"] == "image_generation_diffusion" and unit == "images":
            if not float(megapixels) > 0:
                raise SaefError("megapixels must be > 0")
            n *= max(1.0, float(megapixels) / MEGAPIXELS_1024)
        elif megapixels is not None:
            warnings.append("megapixels applies to per-image diffusion models only and was ignored")
        components.append(_component(row, unit, count, n, p))

    tot = lambda key: sum(c[key] for c in components)
    server, fac = tot("server_wh_central"), tot("facility_wh_central")
    grid = reg["grid_kg_co2e_per_kwh"]
    kg = {}
    for k in CASES:
        kg[k] = tot(f"facility_wh_{k}") / 1e3 * grid + (tot(f"embodied_g_co2e_{k}") + tot(f"construction_g_co2e_{k}")) / 1e3
    high_case = tot("facility_wh_high_case") / 1e3 * grid + (
        tot("embodied_g_co2e_high_case") + tot("construction_g_co2e_high_case")) / 1e3
    water = {}
    for k in CASES:
        water[k] = _calc.water_for_energy(lambda _rid, _p, k=k: (tot(f"server_wh_{k}"), tot(f"facility_wh_{k}")), 1.0, provider, region)
    w = water["central"]
    ml = {k: water[k]["ml_total_reig"] if w["available"] else None for k in CASES}
    types = list(dict.fromkeys(c["model_type"] for c in components))
    classes = list(dict.fromkeys(c["class"] for c in components))
    if mm["model_type"] != "text" and any(t in mm["model_type"].split(";") for t in types):
        warnings += mm["warnings"]
        basis = mm["basis"]
    else:
        warnings += mm["warnings"] if mm["is_default_fallback"] else []
        basis = "billing unit"
    token_units = [c["count"] for c in components if c["unit"].endswith("_tokens")]
    return ModalityEstimate(
        kg_co2e=kg["central"], kg_co2e_low=kg["low"], kg_co2e_high=kg["high"], kg_co2e_high_case=high_case,
        terms_kg_co2e={
            "electricity": fac / 1e3 * grid, "embodied": tot("embodied_g_co2e_central") / 1e3,
            "construction": tot("construction_g_co2e_central") / 1e3, "training": 0.0,
        },
        energy_wh=fac, energy_wh_low=tot("facility_wh_low"), energy_wh_high=tot("facility_wh_high"),
        gpu_energy_wh=tot("gpu_wh_central"), server_energy_wh=server,
        water_ml=ml["central"], water_ml_low=ml["low"], water_ml_high=ml["high"],
        water_ml_excl_hydro=w["ml_total_excl_hydro"] if w["available"] else None,
        model=model, model_type=";".join(types), model_class=";".join(classes), class_basis=basis,
        region_id=reg["region_id"], grid_kg_co2e_per_kwh=grid,
        usage=dict(counts, **({"images": floor_images} if floor_images is not None else {})),
        components=components, warnings=list(dict.fromkeys(warnings + w["warnings"])), provenance=p + w["provenance"],
        water=w, version=version, billed_tokens=sum(token_units) if token_units else None,
    )


VALUE_COLUMNS = [f"{q}_{k}" for q in ("gpu_wh", "server_wh", "facility_wh", "embodied_g_co2e", "construction_g_co2e") for k in CASES] + [
    "facility_wh_high_case", "embodied_g_co2e_high_case", "construction_g_co2e_high_case"]


def _component(row, unit, count, n, p):
    for k in CASES:
        prov(p, f"{row['model_type']} {row['class']} GPU Wh per {row['functional_unit']} ({k})", row, f"gpu_wh_{k}")
    out = {c: float(row[c]) * n for c in VALUE_COLUMNS}
    out.update(model_type=row["model_type"], unit=unit, count=count, functional_unit=row["functional_unit"],
               basis=row["basis"], source_ids=row["source_ids"])
    out["class"] = row["class"]
    return out


def _image_tokens(row, tokens, images, warnings, p):
    """Token-billed images never fall below the smallest open diffusion model ML.ENERGY measured (D-small rows)."""
    key = (row["model_type"], row["class"], "1k_image_output_tokens" if images is None else "1k_image_output_tokens_parity")
    by_tokens = _component(_rows()[key], "image_output_tokens", tokens, tokens / 1e3, p)
    if images is None:
        warnings.append(
            "image count unknown: the central value assumes 1,056 image tokens per image for the diffusion floor and "
            "charges at least one image; pass images for a tighter value"
        )
        images = 1.0 if tokens else 0.0
    floor_row = _rows()["image_generation_diffusion", "D-small", "image"]
    floor = _component(floor_row, "images", images, images, [])
    out = dict(by_tokens)
    for k in CASES + ("high_case",):
        src = "central" if k == "high_case" else k
        if floor[f"gpu_wh_{src}"] > by_tokens[f"gpu_wh_{src}"]:
            out.update({c: floor[c] for c in VALUE_COLUMNS if c.endswith("_" + k)})
            if k == "central":
                out["floored_to_diffusion"] = True
                prov(p, "image_generation_diffusion D-small GPU Wh per image (floor)", floor_row, "gpu_wh_central")
    return out


def usage_from_tokens(mm, model, tokens, output_tokens, input_tokens, cached_input_tokens):
    """Map a plain token split onto the usage units of a non-text model type."""
    types = set(mm["model_type"].split(";"))
    if types & INPUT_ONLY:
        if tokens is None and output_tokens:
            raise SaefError(f"{model!r} (model type {mm['model_type']}) has no output tokens to charge; pass input_tokens")
        return {"input_tokens": tokens if tokens is not None else input_tokens + cached_input_tokens}
    if "moderation" in types and tokens is not None:
        return {"input_tokens": tokens}
    unit = next((OUTPUT_UNIT[t] for t in types if t in OUTPUT_UNIT), None)
    if unit is None or _pick(types, mm["model_class"], "input_tokens") is None:
        have = sorted(n for n in UNITS if _pick(types, mm["model_class"], n) is not None)
        raise SaefError(f"{model!r} (model type {mm['model_type']}) is not charged on tokens; pass usage with one of {', '.join(have)}")
    if tokens is not None:
        raise SaefError(f"{model!r} (model type {mm['model_type']}) needs the input/output token split or usage=, not a billed total")
    if cached_input_tokens and _pick(types, mm["model_class"], "cached_input_tokens") is None:
        input_tokens, cached_input_tokens = input_tokens + cached_input_tokens, 0
    split = {"input_tokens": input_tokens, "cached_input_tokens": cached_input_tokens, unit: output_tokens}
    return {u: v for u, v in split.items() if v} or {"input_tokens": 0}
