import argparse
import json

from . import __version__, citation, estimate, factor, regions
from ._core import DEFAULT_SCENARIO, SaefError


def _src(s):
    return f"[{s['file']} / {s['column']} / {s['row']}]"


def _print_estimate(e):
    print(f"SAEF {e.version}  model={e.model or '-'}  class={e.model_class}  region={e.region}  scenario={e.scenario}")
    if e.reasoning_multiplier != 1:
        print(f"tokens: {e.tokens:,.0f} x reasoning multiplier {e.reasoning_multiplier:g} = {e.effective_tokens:,.0f}")
    else:
        print(f"tokens: {e.tokens:,.0f}")
    print(f"kg CO2e: {e.kg_co2e:.6g}")
    for term, value in e.terms_kg_co2e.items():
        print(f"  {term}: {value:.6g} kg  {_src(e.sources[term])}")
    if e.water_ml is None:
        print("water: not available for this class and region")
    else:
        print(f"water mL: {e.water_ml:.6g}  {_src(e.sources['water'])}")
    if "model_class" in e.sources:
        print(f"model class from: {_src(e.sources['model_class'])}")
    print(e.note)


def main(argv=None):
    p = argparse.ArgumentParser(prog="saef", description="SOMA AI Emission Factors (estimates, not measurements)")
    p.add_argument("--version", action="version", version=f"saef {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="estimate kg CO2e and water for a token count")
    e.add_argument("--tokens", type=float, required=True, help="output tokens")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--model-class", dest="model_class")
    e.add_argument("--region", default="global-average")
    e.add_argument("--scenario", default=DEFAULT_SCENARIO)
    e.add_argument("--reasoning-multiplier", type=float, default=None)
    e.add_argument("--json", action="store_true")

    f = sub.add_parser("factor", help="show the factor for a model class and region")
    f.add_argument("--model-class", dest="model_class", default="B")
    f.add_argument("--region", default="us-east-1")
    f.add_argument("--scenario", default=DEFAULT_SCENARIO)

    sub.add_parser("regions", help="list region ids")
    sub.add_parser("cite", help="print BibTeX")

    args = p.parse_args(argv)
    try:
        if args.cmd == "estimate":
            res = estimate(
                args.tokens,
                model=args.model,
                model_class=args.model_class,
                region=args.region,
                scenario=args.scenario,
                reasoning_multiplier=args.reasoning_multiplier,
            )
            if args.json:
                print(json.dumps(res.to_dict(), indent=2))
            else:
                _print_estimate(res)
        elif args.cmd == "factor":
            print(json.dumps(factor(args.model_class, args.region, args.scenario), indent=2))
        elif args.cmd == "regions":
            print("\n".join(regions()))
        else:
            print(citation())
    except SaefError as err:
        print(f"saef: error: {err}")
        return 2
    return 0
