import argparse
import json

from . import __version__, citation, estimate, estimate_modality, factor, regions
from ._core import ModalityEstimate, SaefError


def _print_estimate(e):
    print(f"SAEF {e.version}  model={e.model or '-'}  class={e.model_class} ({e.class_basis})  region={e.region_id}  grid={e.grid_kg_co2e_per_kwh:g} kg/kWh")
    print(f"billed tokens: {e.billed_tokens:,.0f}  effective tokens: {e.effective_tokens:,.0f}")
    print(f"kg CO2e: {e.kg_co2e:.6g}   (conservative high case {e.kg_co2e_high_case:.6g}; development line {e.development_line_kg_co2e:.6g}, reported separately)")
    for term, value in e.terms_kg_co2e.items():
        print(f"  {term}: {value:.6g} kg")
    if e.water_ml is None:
        print("water: not available for this region")
    else:
        print(f"water mL: {e.water_ml:.6g} (hydropower allocated), {e.water_ml_excl_hydro:.6g} (excluding hydropower)")
    for w in e.warnings:
        print(f"warning: {w}")
    print("sources (file:line, column):")
    for s in e.provenance:
        print(f"  {s['quantity']}: {s['file']}:{s['line']} {s['column']} = {s['value']}")
    print(e.note)


def _print_modality(e):
    print(f"SAEF {e.version}  model={e.model or '-'}  type={e.model_type}  class={e.model_class} ({e.class_basis})  region={e.region_id}  grid={e.grid_kg_co2e_per_kwh:g} kg/kWh")
    print("usage: " + ", ".join(f"{k}={v:,g}" for k, v in e.usage.items()))
    print(f"facility energy Wh: {e.energy_wh:.6g} ({e.energy_wh_low:.6g} to {e.energy_wh_high:.6g})")
    print(f"kg CO2e: {e.kg_co2e:.6g} ({e.kg_co2e_low:.6g} to {e.kg_co2e_high:.6g}; conservative overhead case {e.kg_co2e_high_case:.6g})")
    for term, value in e.terms_kg_co2e.items():
        print(f"  {term}: {value:.6g} kg")
    if e.water_ml is None:
        print("water: not available for this region")
    else:
        print(f"water mL: {e.water_ml:.6g} ({e.water_ml_low:.6g} to {e.water_ml_high:.6g}, hydropower allocated), {e.water_ml_excl_hydro:.6g} (excluding hydropower)")
    for w in e.warnings:
        print(f"warning: {w}")
    print("sources (file:line, column):")
    for s in e.provenance:
        print(f"  {s['quantity']}: {s['file']}:{s['line']} {s['column']} = {s['value']}")
    print(e.note)


def main(argv=None):
    p = argparse.ArgumentParser(prog="saef", description="SOMA AI Emission Factors (estimates, not measurements)")
    p.add_argument("--version", action="version", version=f"saef {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="estimate kg CO2e and water for a token count")
    e.add_argument("--tokens", type=float, help="billed total (input plus output)")
    e.add_argument("--output-tokens", type=float, help="output tokens, with --input-tokens")
    e.add_argument("--input-tokens", type=float, default=0.0)
    e.add_argument("--cached-input-tokens", type=float, default=0.0)
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--model-class", dest="model_class")
    e.add_argument("--region", default=None, help="cloud region id, ISO-3 country, eGRID subregion, eu-27; default global average")
    e.add_argument("--provider", default=None)
    e.add_argument("--json", action="store_true")

    m = sub.add_parser("modality", help="estimate an embedding, rerank, moderation, image, transcription, speech, voice or video usage")
    m.add_argument("--model")
    m.add_argument("--model-type", dest="model_type", help="with --model-class, instead of --model")
    m.add_argument("--model-class", dest="model_class")
    m.add_argument("--unit", required=True, help="for example input_tokens, image_output_tokens, images, audio_minutes, search_units, calls, characters, conversation_minutes, video_seconds_1080p")
    m.add_argument("--count", type=float, required=True)
    m.add_argument("--megapixels", type=float, default=None, help="per-image diffusion models above 1024x1024")
    m.add_argument("--region", default=None)
    m.add_argument("--provider", default=None)
    m.add_argument("--json", action="store_true")

    f = sub.add_parser("factor", help="show the class factor at a region")
    f.add_argument("--model-class", dest="model_class", default="B")
    f.add_argument("--region", default="us-east-1")
    f.add_argument("--provider", default=None)

    r = sub.add_parser("regions", help="list cloud region ids")
    r.add_argument("--provider", default=None)
    sub.add_parser("cite", help="print BibTeX")

    args = p.parse_args(argv)
    try:
        if args.cmd == "estimate":
            res = estimate(
                tokens=args.tokens, model=args.model, model_class=args.model_class, region=args.region,
                provider=args.provider, output_tokens=args.output_tokens, input_tokens=args.input_tokens,
                cached_input_tokens=args.cached_input_tokens,
            )
            if args.json:
                print(json.dumps(res.to_dict(), indent=2))
            elif isinstance(res, ModalityEstimate):
                _print_modality(res)
            else:
                _print_estimate(res)
        elif args.cmd == "modality":
            res = estimate_modality(
                model=args.model, unit_count=args.count, unit=args.unit, region=args.region, provider=args.provider,
                megapixels=args.megapixels, model_type=args.model_type, model_class=args.model_class,
            )
            if args.json:
                print(json.dumps(res.to_dict(), indent=2))
            else:
                _print_modality(res)
        elif args.cmd == "factor":
            print(json.dumps(factor(args.model_class, args.region, args.provider), indent=2))
        elif args.cmd == "regions":
            print("\n".join(regions(args.provider)))
        else:
            print(citation())
    except SaefError as err:
        print(f"saef: error: {err}")
        return 2
    return 0
