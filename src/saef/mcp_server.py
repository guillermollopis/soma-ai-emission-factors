"""MCP server exposing SAEF estimates to AI agents over stdio. Needs the optional `mcp` extra."""
from mcp.server.fastmcp import FastMCP

from . import DATASET_VERSION, SaefError, __version__, estimate, model_class_for
from . import _calc

EXCHANGE_INPUT_TOKENS = 928
EXCHANGE_OUTPUT_TOKENS = 282
EXCHANGE_TOKENS = EXCHANGE_INPUT_TOKENS + EXCHANGE_OUTPUT_TOKENS

CLASS_DESCRIPTIONS = {
    "A": "small model (for example mini, nano, flash or 8B to 14B open-weight models)",
    "B": "mid-size model (for example GPT-4o, Claude Sonnet, Gemini Pro)",
    "C": "large or frontier model (for example the largest GPT, Claude Opus or Gemini Ultra tiers)",
}

DOI = "10.5281/zenodo.20443585"
ARXIV = "2606.10660"
PAPER_DOI = "10.1016/j.spc.2026.11.003"
HOMEPAGE = "https://somaai.earth/ai-footprint"
NOTE = "Estimate, not a measurement. SOMA AI Emission Factors (SAEF) factors are modelled from published benchmarks and grid data."

mcp = FastMCP("saef")


def _check_tokens(tokens):
    if tokens < 0:
        raise ValueError("tokens must be >= 0")


def _report(e):
    grid = e.grid_kg_co2e_per_kwh
    out = {
        "carbon": {
            "kg_co2e": e.kg_co2e,
            "g_co2e": e.kg_co2e * 1000,
            "high_case_kg_co2e": e.kg_co2e_high_case,
            "high_case_g_co2e": e.kg_co2e_high_case * 1000,
            "note": "kg_co2e is the central case. The high case is a conservative upper bound.",
        },
        "model": e.model,
        "model_class": e.model_class,
        "class_basis": e.class_basis,
        "region_used": e.region_id,
        "grid_kg_co2e_per_kwh": grid,
        "billed_tokens": e.billed_tokens,
        "dataset_version": e.version,
        "warnings": e.warnings,
        "provenance": e.provenance,
        "citation": f"SOMA AI Emission Factors (SAEF) v{e.version}, doi:{DOI}; method paper doi:{PAPER_DOI}",
        "note": NOTE,
    }
    if grid > 0:
        out["energy"] = {
            "facility_wh": e.terms_kg_co2e["electricity"] / grid * 1000,
            "note": "Facility-level electricity including cooling and overhead, derived from the electricity term and the grid intensity.",
        }
    if e.water_ml is not None:
        out["water"] = {
            "ml": e.water_ml,
            "ml_excluding_hydropower_allocation": e.water_ml_excl_hydro,
            "note": "Two hydropower conventions exist and neither is preferred. ml allocates reservoir evaporation to electricity.",
        }
    else:
        out["water"] = {"ml": None, "note": "No water factor for this region. Water is not available, not zero."}
    return out


@mcp.tool()
def estimate_footprint(tokens: int, model: str, region: str = "global") -> dict:
    """Estimate carbon, energy and water for AI inference using SOMA AI Emission Factors (SAEF).

    Args:
        tokens: Billed tokens, input plus output, as shown on the provider invoice or usage page.
        model: Model identifier such as "gpt-4o", "claude-sonnet-4-5" or "gemini-2.5-flash". Unmapped models are treated as the conservative class C.
        region: Cloud region id (for example "eu-west-1", "us-east-1"), ISO-3 country code (for example "ESP"), eGRID subregion, "eu-27" or "global" for the world-average grid. Use list_regions to see options.

    Returns carbon in kg and g CO2e with a central and a high case, facility energy in Wh, water in mL where the region has a water factor, the model class (A small, B mid-size, C large), the region used, the dataset version and a provenance trail to the source CSV rows.
    All figures are estimates, not measurements. They exclude model training development lines and are intended for Scope 3 category 1 reporting.
    """
    _check_tokens(tokens)
    return _report(estimate(tokens=tokens, model=model, region=region))


@mcp.tool()
def estimate_chat_usage(messages: int, model: str, region: str = "global") -> dict:
    """Estimate carbon, energy and water for chat usage, counted in messages, using SOMA AI Emission Factors (SAEF).

    Each message is one chat exchange (a question and its answer) of 1,210 tokens, the dataset's short-chat default of 928 input and 282 output tokens. Long documents, coding agents and reasoning modes use far more tokens, so use estimate_footprint when you know the token count.

    Args:
        messages: Number of chat exchanges.
        model: Model identifier such as "gpt-4o" or "claude-sonnet-4-5".
        region: Cloud region id, ISO-3 country code, eGRID subregion, "eu-27" or "global". Use list_regions to see options.

    Same output as estimate_footprint, plus the token assumption used. All figures are estimates, not measurements.
    """
    if messages < 0:
        raise ValueError("messages must be >= 0")
    e = estimate(
        model=model, region=region,
        input_tokens=EXCHANGE_INPUT_TOKENS * messages, output_tokens=EXCHANGE_OUTPUT_TOKENS * messages,
    )
    out = _report(e)
    out["messages"] = messages
    out["assumption"] = (
        f"{EXCHANGE_TOKENS} tokens per exchange ({EXCHANGE_INPUT_TOKENS} input, {EXCHANGE_OUTPUT_TOKENS} output), "
        "the SAEF short-chat default"
    )
    return out


@mcp.tool()
def model_class(model: str) -> dict:
    """Return the SAEF model class (A small, B mid-size, C large) for a model identifier.

    SOMA AI Emission Factors (SAEF) assign factors by class, not by brand. A model that matches no known pattern gets the conservative class C. Embeddings, image, audio, video and moderation models are out of scope and return an error.
    """
    cls = model_class_for(model)
    return {"model": model, "model_class": cls, "description": CLASS_DESCRIPTIONS[cls], "dataset_version": DATASET_VERSION}


@mcp.tool()
def list_regions() -> dict:
    """List the region codes accepted by the other SAEF tools.

    Returns the special values ("global", "eu-27"), the cloud regions of AWS, Azure and Google Cloud with names, and how to pass a country. Any ISO-3 country code (for example "ESP", "DEU", "USA") or eGRID subregion (for example "RFCE") also works.
    """
    cloud = [
        {"region": r["region_id"], "provider": r["provider"], "name": r["region_name"]}
        for r in _calc.read("cloud_region_factors.csv")
    ]
    cloud.sort(key=lambda r: (r["provider"], r["region"]))
    countries = {r["grid_id"]: r["name"] for r in _calc.read("grid_intensity_all.csv") if r["kind"] == "country"}
    return {
        "special": {"global": "world-average grid", "eu-27": "EU-27 average grid"},
        "cloud_regions": cloud,
        "country_codes": countries,
        "dataset_version": DATASET_VERSION,
    }


@mcp.tool()
def compare_regions(tokens: int, model: str, regions: list[str]) -> dict:
    """Compare the carbon and water estimate for the same workload across regions using SOMA AI Emission Factors (SAEF).

    Args:
        tokens: Billed tokens, input plus output.
        model: Model identifier such as "gpt-4o".
        regions: Region codes to compare, for example ["eu-north-1", "us-east-1", "ap-south-1"]. See list_regions.

    Returns one row per region, sorted from lowest to highest carbon, with kg and g CO2e (central and high case) and water in mL. The lowest-carbon grid is not always the lowest-water one. A region that is not recognised gets an error entry instead of failing the whole call. Estimates, not measurements.
    """
    _check_tokens(tokens)
    rows = []
    errors = []
    cls = None
    for region in regions:
        try:
            e = estimate(tokens=tokens, model=model, region=region)
        except SaefError as exc:
            errors.append({"region": region, "error": str(exc)})
            continue
        cls = e.model_class
        rows.append({
            "region": region,
            "region_used": e.region_id,
            "kg_co2e": e.kg_co2e,
            "g_co2e": e.kg_co2e * 1000,
            "high_case_kg_co2e": e.kg_co2e_high_case,
            "water_ml": e.water_ml,
        })
    rows.sort(key=lambda r: r["kg_co2e"])
    return {
        "tokens": tokens,
        "model": model,
        "model_class": cls,
        "regions": rows,
        "errors": errors,
        "dataset_version": DATASET_VERSION,
        "note": NOTE,
    }


@mcp.resource("saef://about")
def about() -> str:
    """About SAEF, its licence, citation and links."""
    return (
        f"SOMA AI Emission Factors (SAEF), package saef {__version__}, dataset v{DATASET_VERSION}.\n"
        "Open, versioned estimates of carbon (kg CO2e), energy and water for AI inference, per token, by model class and region, "
        "for Scope 3 category 1 reporting. All figures are estimates, not measurements.\n"
        "Licence: data SOMA AI Emission Factors Data Licence 1.0 (Non-Commercial), code PolyForm Noncommercial 1.0.0. "
        "Free for personal use, academic and non-profit research, journalism, charities and government. Any use by a company or freelancer, including internal ESG/CSRD reporting, needs a commercial licence: https://somaai.earth/licensing (guillermollopis@somaai.earth). Versions up to v3.2.0 remain CC BY 4.0 (data) and MIT (code).\n"
        f"Dataset DOI: {DOI} (https://doi.org/{DOI})\n"
        f"Method paper: Sustainable Production and Consumption 70 (2026) 320-347, doi {PAPER_DOI} (https://doi.org/{PAPER_DOI}), preprint arXiv {ARXIV}\n"
        f"Homepage: {HOMEPAGE}\n"
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
