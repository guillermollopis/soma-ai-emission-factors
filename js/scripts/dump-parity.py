"""Dump expected values from the Python package for the TypeScript parity tests.

Usage: PYTHONPATH=path/to/saef-repo/src python scripts/dump-parity.py
"""
import csv
import json
import sys
from pathlib import Path

import saef
from saef import _calc
from saef.integrations.openai import estimate_from_response

out_path = Path(__file__).resolve().parent.parent / "test" / "parity.json"

measured = sorted({r["model_id"] for r in _calc.read("model_factors.csv")})[:6]
mapping_fix = ["gpt-4o-mini-transcribe-2025-12-15", "gpt-4o-mini-tts", "TitanEmbeddingV2-Text", "advanced-voice-4o", "DeepSeekV3"]
models = [
    "gpt-4o", "gpt-4o-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "o3", "claude-sonnet-4-5", "claude-haiku-4-5",
    "claude-opus-4-1", "gemini-2.5-pro", "gemini-2.5-flash", "mistral-large-latest", "llama-3.1-70b-instruct",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "openai/gpt-4o", "azureml://registries/azure-openai/models/gpt-4o/versions/2",
    "totally-unknown-model-xyz", "text-embedding-3-small", "dall-e-3",
] + mapping_fix + measured
regions = [None, "us-east-1", "eu-north-1", "eu-west-1", "global-average", "eu-27", "swedencentral", "FRA", "DEU", "us-west-2", "ap-southeast-2", "nope-region-9"]
splits = [
    dict(input_tokens=1000, output_tokens=500, cached_input_tokens=0),
    dict(input_tokens=200, output_tokens=150, cached_input_tokens=1800),
    dict(input_tokens=0, output_tokens=1, cached_input_tokens=0),
    dict(input_tokens=2_500_000, output_tokens=400_000, cached_input_tokens=1_000_000),
]



def text_estimate(model, **kw):
    """The TypeScript package covers text models only."""
    if saef.model_type_for(model)[0] != "text":
        raise saef.SaefError("not a text model")
    return saef.estimate(model=model, **kw)


cases = []
for m in models:
    for rg in regions:
        for sp in splits:
            c = dict(kind="split", model=m, region=rg, **sp)
            try:
                e = text_estimate(model=m, region=rg, **sp)
                c["expected"] = dict(
                    kg_co2e=e.kg_co2e, kg_co2e_high_case=e.kg_co2e_high_case,
                    development_line_kg_co2e=e.development_line_kg_co2e, terms=e.terms_kg_co2e,
                    kwh=e.terms_kg_co2e["electricity"] / e.grid_kg_co2e_per_kwh * 1 if e.grid_kg_co2e_per_kwh else None,
                    water_ml=e.water_ml, water_ml_excl_hydro=e.water_ml_excl_hydro,
                    model_class=e.model_class, class_basis=e.class_basis, region_id=e.region_id,
                    grid=e.grid_kg_co2e_per_kwh, billed=e.billed_tokens, effective=e.effective_tokens, warnings=e.warnings,
                )
            except saef.SaefError:
                c["expected"] = None
            cases.append(c)
        for toks in (1_000_000, 12345):
            c = dict(kind="billed", model=m, region=rg, tokens=toks)
            try:
                e = text_estimate(model=m, region=rg, tokens=toks)
                c["expected"] = dict(
                    kg_co2e=e.kg_co2e, kg_co2e_high_case=e.kg_co2e_high_case, water_ml=e.water_ml,
                    model_class=e.model_class, region_id=e.region_id, effective=e.effective_tokens, billed=e.billed_tokens,
                )
            except saef.SaefError:
                c["expected"] = None
            cases.append(c)

classes = []
for cl in "ABC":
    for rg in ["us-east-1", "eu-north-1", "eu-west-1", "global-average", "FRA"]:
        classes.append(dict(model_class=cl, region=rg, factor=saef.factor(cl, rg if rg != "global-average" else None)["kg_co2e_per_1m_billed_tokens"]))

resp_cases = []
shapes = {
    "chat": {"model": "gpt-4o", "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 800}}},
    "chat_nocache": {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 50, "completion_tokens": 20}},
    "responses": {"object": "response", "model": "gpt-5", "usage": {"input_tokens": 1500, "output_tokens": 400, "input_tokens_details": {"cached_tokens": 1000}}},
    "responses_nodetail": {"object": "response", "model": "gpt-4.1-mini", "usage": {"input_tokens": 90, "output_tokens": 40}},
    "anthropic": {"type": "message", "model": "claude-sonnet-4-5", "usage": {"input_tokens": 200, "output_tokens": 150, "cache_read_input_tokens": 1800, "cache_creation_input_tokens": 0}},
    "anthropic_write": {"type": "message", "model": "claude-haiku-4-5", "usage": {"input_tokens": 300, "output_tokens": 80, "cache_read_input_tokens": 500, "cache_creation_input_tokens": 700}},
}
for name, resp in shapes.items():
    for rg in [None, "eu-west-1", "us-east-1", "eu-north-1"]:
        e = estimate_from_response(resp, region=rg)
        resp_cases.append(dict(name=name, response=resp, region=rg, expected=dict(
            kg_co2e=e.kg_co2e, kg_co2e_high_case=e.kg_co2e_high_case, water_ml=e.water_ml,
            model_class=e.model_class, region_id=e.region_id, billed=e.billed_tokens, effective=e.effective_tokens)))

classmap = {m: (saef.model_class_for(m) if saef.model_type_for(m)[0] == "text" else None) for m in models}
out = dict(
    version=saef.__version__, dataset_version=saef.DATASET_VERSION, cases=cases, factors=classes, responses=resp_cases,
    model_class=classmap, regions=saef.regions(), grids=saef.grids(), citation=saef.citation(),
)
out_path.write_text(json.dumps(out, indent=None, allow_nan=False), encoding="utf-8")
print(out_path, len(cases), len(resp_cases), out_path.stat().st_size)
