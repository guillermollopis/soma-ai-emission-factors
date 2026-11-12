"""Calculator for the SAEF v3 lifecycle-boundary tables. Reads only the bundled CSVs.

Method (dataset V3_LIFECYCLE_SPEC.md, section 10):
    kg CO2e = T_eff / 1e6 x inference footprint + T_billed / 1e6 x training term
    T_eff   = (T_out + W_in x T_in_uncached + W_cache x T_in_cached) / D
Never multiply a billed token count by an inference footprint.
"""
import csv
import difflib
import io
import re
from functools import lru_cache
from importlib import resources

PATH_NOISE = {
    "models", "model", "publishers", "registries", "versions", "version", "projects", "locations",
    "deployments", "inference-profile", "inference-profiles", "endpoints", "foundry", "openai",
    "azureml:", "azureml", "arn:aws:bedrock:", "providers", "catalogs", "assets",
}
DOT_REGION = {"us", "eu", "apac", "us-gov", "global", "jp", "au", "ca", "sa", "me", "il", "uk", "in"}
DOT_VENDOR = {
    "anthropic", "amazon", "meta", "mistral", "cohere", "ai21", "stability", "deepseek", "openai",
    "google", "writer", "luma", "twelvelabs", "qwen", "nvidia", "moonshot", "zai", "minimax",
    "microsoft", "xai", "alibaba", "tencent", "baidu", "perplexity", "snowflake", "databricks",
}
UNKNOWN_REGION = {"", "unknown", "global", "global-average", "world", "any", "none", "mixed", "multi-region"}
EU27_REGION = {"eu-27", "eu27", "eu", "eu-only", "eu data zone", "european union"}


class SaefError(ValueError):
    pass


@lru_cache(maxsize=None)
def read(name):
    text = resources.files("saef").joinpath("data", name).read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    for i, r in enumerate(rows, 2):
        r["_file"], r["_line"] = name, i
    return tuple(rows)


def prov(out, quantity, row, col):
    out.append({"quantity": quantity, "file": row["_file"], "line": row["_line"], "column": col, "value": row[col]})


@lru_cache(maxsize=None)
def constants():
    return {r["constant"]: r for r in read("token_basis_constants.csv")}


def const(name):
    return float(constants()[name]["central"])


def normalise_identifier(identifier):
    s = str(identifier).strip().lower()
    cands = [s]
    body = s
    if body.startswith("arn:aws:bedrock:"):
        body = body.rsplit("/", 1)[-1]
    segs = [x for x in body.split("/") if x and x not in PATH_NOISE and not x.isdigit()]
    if segs:
        body = segs[-1]
    cands.append(body)
    stripped, dropped = body, []
    while "." in stripped:
        head, rest = stripped.split(".", 1)
        if head in DOT_REGION or head in DOT_VENDOR:
            dropped.append(head)
            stripped = rest
        else:
            break
    if stripped != body:
        cands.append(stripped)
        if dropped and dropped[-1] in DOT_VENDOR:
            cands.append(f"{dropped[-1]}-{stripped}")
    cands += [c.replace(".", "-") for c in cands if "." in c]
    out = list(dict.fromkeys(c for c in cands if c))
    return out


def _rank(pattern):
    lit_prefix = pattern.split("*", 1)[0]
    return (len(lit_prefix), len(pattern.replace("*", "")), len(pattern), pattern)


@lru_cache(maxsize=None)
def _patterns():
    out = []
    for r in read("model_class_mapping.csv"):
        for alt in r["pattern"].split(";"):
            alt = alt.strip().lower()
            if alt:
                rx = re.compile("".join(".*" if p == "*" else re.escape(p) for p in re.split(r"(\*)", alt)))
                out.append((_rank(alt), rx, r))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _measured_id(cands):
    byid = {r["model_id"].lower(): r["model_id"] for r in read("model_factors.csv")}
    bylast = {k.rsplit("/", 1)[-1]: v for k, v in byid.items()}
    for c in cands:
        c = c.strip().lower()
        if c in byid:
            return byid[c]
        if c in bylast:
            return bylast[c]
    return None


def map_model(identifier):
    cands = normalise_identifier(identifier)
    warnings, p = [], []
    hit = next((r for _k, rx, r in _patterns() if any(rx.match(c) for c in cands)), None)
    if hit is None:
        raise SaefError("model_class_mapping.csv has no default row")
    prov(p, "model class", hit, "model_class")
    out = dict(
        identifier=str(identifier), pattern=hit["pattern"], basis=hit["basis"], model_class=hit["model_class"],
        own_factor_model_id=hit["own_factor_model_id"] or None, out_of_scope=hit["model_class"] == "out-of-scope",
        is_default_fallback=hit["basis"] == "default", warnings=warnings, provenance=p,
    )
    if out["is_default_fallback"]:
        warnings.append(f"unmapped model identifier {identifier!r}: defaulted to Class {hit['model_class']} and flagged")
    if out["out_of_scope"]:
        out["model_class"] = None
        warnings.append(f"{identifier!r} matches the out-of-scope pattern {hit['pattern']!r}: {hit['notes']}")
    elif hit["basis"] == "tier":
        warnings.append(
            f"class proxy for a closed model: {identifier!r} was placed in Class {hit['model_class']} by the "
            f"vendor pricing tier, not by a disclosed size ({hit['size_or_tier_note']})"
        )
    direct = _measured_id(cands + [str(identifier)])
    if direct:
        out["own_factor_model_id"] = direct
        out["basis"] = "measured"
    return out


def _suggest(region):
    pool = {r["region_id"] for r in read("cloud_region_factors.csv")} | {r["grid_id"] for r in read("grid_intensity_all.csv")}
    low = region.lower()
    near = sorted(x for x in pool if x.lower().startswith(low))[:8] or difflib.get_close_matches(region, sorted(pool), n=5)
    return near


def resolve_region(provider=None, region=None):
    """Resolve to one grid row. Unknown or global regions use the global average, as the dataset's decision rule says;
    a region string that matches nothing raises SaefError."""
    warnings, p = [], []
    grids = {r["grid_id"]: r for r in read("grid_intensity_all.csv")}
    key = (region or "").strip()
    low = key.lower()

    def finish(grid_row, how, class_row=None, region_id=None):
        prov(p, "grid carbon intensity", grid_row, "grid_kg_co2e_per_kwh")
        return dict(
            grid_id=grid_row["grid_id"], grid_name=grid_row["name"],
            grid_kg_co2e_per_kwh=float(grid_row["grid_kg_co2e_per_kwh"]),
            class_row=grid_row if class_row is None else class_row,
            region_id=region_id or grid_row["grid_id"], how=how, warnings=warnings, provenance=p,
        )

    if low in UNKNOWN_REGION:
        warnings.append("region unknown or global: the global-average grid row (WORLD) is applied, per the dataset's serving-region decision rule")
        return finish(grids["WORLD"], "unknown -> global average", region_id="global-average")
    if low in EU27_REGION:
        warnings.append("EU data zone or EU-only processing: the EU-27 average grid row (EU27) is applied")
        return finish(grids["EU27"], "EU-only processing -> EU-27", region_id="eu-27")

    cloud = [r for r in read("cloud_region_factors.csv") if r["region_id"].lower() == low]
    if provider:
        cloud = [r for r in cloud if r["provider"].lower() == provider.strip().lower()]
    if cloud:
        if len({r["grid_id"] for r in cloud}) > 1:
            raise SaefError(f"Region {region!r} is used by several providers with different grids, pass provider=")
        r = cloud[0]
        if r["mapping_confidence"] == "medium":
            warnings.append(f"region mapping confidence is medium for {r['provider']}/{r['region_id']}: {r['mapping_notes']}")
        prov(p, "cloud region to grid", r, "grid_id")
        return finish(grids[r["grid_id"]], f"cloud region {r['provider']}/{r['region_id']}", class_row=r, region_id=r["region_id"])

    hit = next((r for r in read("region_grid_crosswalk.csv") if r["region_id"].lower() == low), None)
    if hit:
        prov(p, "printed region to grid", hit, "grid_id")
        return finish(grids[hit["grid_id"]], f"printed region {hit['region_id']}", region_id=hit["region_id"])
    if key.upper() in grids:
        g = grids[key.upper()]
        return finish(g, f"{g['kind'].replace('_', ' ')} {g['grid_id']}")
    byname = [r for r in grids.values() if r["name"].strip().lower() == low]
    if len(byname) == 1:
        return finish(byname[0], f"country name -> {byname[0]['grid_id']}")
    hint = _suggest(key)
    msg = f"Unknown region {region!r}."
    if hint:
        msg += f" Did you mean: {', '.join(hint)}?"
    msg += " Use a cloud region id (see saef.regions()), an ISO-3 country code, an eGRID subregion, 'eu-27' or 'unknown'."
    raise SaefError(msg)


def effective_tokens(output=0.0, uncached_input=0.0, cached_input=0.0):
    return (output + const("W_in") * uncached_input + const("W_cache") * cached_input) / const("D")


def effective_from_billed(billed):
    s = const("default_input_share")
    return billed * (1 - s + const("W_in") * s) / const("D")


def _class_terms(cl, reg, p):
    c = cl.lower()
    row = reg["class_row"]
    cols = {
        "footprint": f"class_{c}_inference_footprint_kg_per_m_effective_tokens",
        "training": f"class_{c}_training_kg_per_m_billed_tokens",
        "footprint_high": f"class_{c}_inference_footprint_high_case_kg_per_m_effective_tokens",
        "development": f"class_{c}_development_line_kg_per_m_effective_tokens",
        "electricity": f"class_{c}_electricity_kg_per_m_effective_tokens",
        "embodied": f"class_{c}_embodied_kg_per_m_effective_tokens",
        "construction": f"class_{c}_construction_kg_per_m_effective_tokens",
        "per_billed": f"class_{c}_factor_kg_per_m_billed_tokens_default_mix",
    }
    if cols["footprint"] not in row:
        raise SaefError(f"no class factors for model class {cl!r}; use A, B or C")
    f = {k: float(row[col]) for k, col in cols.items()}
    for k in ("footprint", "training", "footprint_high", "development"):
        prov(p, k.replace("_", " "), row, cols[k])
    f["basis"] = f"class {cl} mean of the measured members, H100"
    return f


def _model_terms(model_id, reg, p):
    r = next((r for r in read("model_factors.csv") if r["model_id"] == model_id and r["gpu"] == "H100"), None)
    if r is None:
        return None
    grid = reg["grid_kg_co2e_per_kwh"]
    fac, fac_hi = float(r["facility_wh_per_1k_effective_tokens"]), float(r["facility_wh_high_case_per_1k_effective_tokens"])
    emb, con = float(r["embodied_kg_co2e_per_m_tokens"]), float(r["construction_kg_co2e_per_m_tokens"])
    emb_hi, con_hi = float(r["embodied_high_case_kg_co2e_per_m_tokens"]), float(r["construction_high_case_kg_co2e_per_m_tokens"])
    tr = float(r["training_kg_co2e_per_m_billed_tokens"])
    for q, col in (("facility Wh per 1,000 effective tokens", "facility_wh_per_1k_effective_tokens"),
                   ("embodied", "embodied_kg_co2e_per_m_tokens"), ("construction", "construction_kg_co2e_per_m_tokens"),
                   ("training term", "training_kg_co2e_per_m_billed_tokens")):
        prov(p, q, r, col)
    mult = (1 - const("default_input_share") + const("W_in") * const("default_input_share")) / const("D")
    foot = fac * grid + emb + con
    return dict(
        footprint=foot, training=tr, footprint_high=fac_hi * grid + emb_hi + con_hi,
        development=float(r["development_line_kg_co2e_per_m_tokens_global"]) / float(r["grid_kg_co2e_per_kwh"]) * grid,
        electricity=fac * grid, embodied=emb, construction=con, per_billed=foot * mult + tr,
        model_class=r["model_class"], basis=f"measured model {model_id}, H100",
    )


def emissions(model=None, model_class=None, provider=None, region=None, billed_tokens=None,
              output_tokens=0.0, input_tokens=0.0, cached_input_tokens=0.0):
    warnings, p = [], []
    if (model is None) == (model_class is None):
        raise SaefError("Pass exactly one of model or model_class")
    own = None
    if model is not None:
        mm = map_model(model)
        warnings += mm["warnings"]
        p += mm["provenance"]
        if mm["out_of_scope"]:
            raise SaefError(f"{model!r} is out of scope (embeddings, image, audio, video or moderation): no estimate. {mm['warnings'][-1]}")
        cl, own, basis = mm["model_class"], mm["own_factor_model_id"], mm["basis"]
        matched = mm["pattern"]
    else:
        cl = {"a": "A", "b": "B", "c": "C"}.get(str(model_class).strip().lower().replace("class_", "")[:1], None)
        if cl is None:
            raise SaefError(f"Unknown model class {model_class!r}; use A, B or C")
        basis, matched = "caller-supplied", None
    reg = resolve_region(provider, region)
    warnings += reg["warnings"]
    p += reg["provenance"]

    if billed_tokens is not None:
        billed = float(billed_tokens)
        t_eff = effective_from_billed(billed)
        warnings.append(
            f"no input/output split given: the default input share {const('default_input_share'):g} and no cache credit were applied"
        )
    else:
        billed = float(input_tokens) + float(cached_input_tokens) + float(output_tokens)
        t_eff = effective_tokens(float(output_tokens), float(input_tokens), float(cached_input_tokens))
        if not cached_input_tokens:
            warnings.append("no cached-token count given, so no cache credit was taken")
    for k in ("W_in", "W_cache", "D", "default_input_share"):
        prov(p, f"token basis {k}", constants()[k], "central")

    f = _model_terms(own, reg, p) if own else None
    if f is not None:
        cl, basis = f["model_class"], "measured"
    else:
        f = _class_terms(cl, reg, p)
    kg = t_eff / 1e6 * f["footprint"] + billed / 1e6 * f["training"]
    kg_high = t_eff / 1e6 * f["footprint_high"] + billed / 1e6 * f["training"]
    return dict(
        kg_co2e=kg, kg_co2e_high_case=kg_high, development_line_kg_co2e=t_eff / 1e6 * f["development"],
        electricity_kg_co2e=t_eff / 1e6 * f["electricity"], embodied_kg_co2e=t_eff / 1e6 * f["embodied"],
        construction_kg_co2e=t_eff / 1e6 * f["construction"], training_kg_co2e=billed / 1e6 * f["training"],
        model=model, matched_pattern=matched, model_class=cl, class_basis=basis, own_factor_model_id=own,
        is_default_fallback=bool(model is not None and mm["is_default_fallback"]),
        region_id=reg["region_id"], grid_id=reg["grid_id"], grid_kg_co2e_per_kwh=reg["grid_kg_co2e_per_kwh"],
        region_resolution=reg["how"], billed_tokens=billed, effective_tokens=t_eff, factors=f,
        warnings=warnings, provenance=p, _reg=reg,
    )


def water(t_eff, model_class, provider=None, region=None):
    """Water for an effective-token count under both hydropower conventions, or available=False."""
    warnings, p = [], []
    reg = resolve_region(provider, region)
    warnings += reg["warnings"]
    rid = reg["region_id"]
    wf = {r["region_id"]: r for r in read("region_water_factors.csv")}
    if reg["how"].startswith("unknown") or rid not in wf:
        warnings.append(
            f"no water factor for region {rid!r}: region_water_factors.csv covers {len(wf)} regions only "
            f"and has no unknown-region fallback. Disclose water as not available"
        )
        return dict(available=False, region_id=rid, warnings=warnings, provenance=p)
    w = wf[rid]
    fl = next((r for r in read("factors_v3_lifecycle.csv") if r["model_class"] == model_class and r["gpu"] == "H100" and r["region_id"] == rid), None)
    if fl is None:
        raise SaefError(f"no factors_v3_lifecycle.csv row for class {model_class} / H100 / {rid}")
    fac, srv = float(fl["facility_wh_per_1k_effective_tokens"]), float(fl["server_wh_per_1k_effective_tokens"])
    prov(p, "facility Wh per 1,000 effective tokens", fl, "facility_wh_per_1k_effective_tokens")
    prov(p, "server Wh per 1,000 effective tokens", fl, "server_wh_per_1k_effective_tokens")
    k = t_eff / 1000.0
    base = srv if w["wue_denominator"].startswith("IT load") else fac
    onsite = base * float(w["wue_l_per_kwh"])
    if w["wue_basis_as_published"] == "withdrawal":
        onsite *= float(w["withdrawal_to_consumption"])
    prov(p, "on-site WUE", w, "wue_l_per_kwh")
    out = dict(available=True, region_id=rid, ml_onsite=onsite * k, ml_offsite_reig=None, ml_offsite_excl_hydro=None,
               ml_total_reig=None, ml_total_excl_hydro=None, warnings=warnings, provenance=p)
    if not w["ewif_l_per_kwh"]:
        warnings.append(f"no off-site water factor for {rid}: {w['ewif_source']}. Only the on-site term is reported")
    else:
        off, off_x = fac * float(w["ewif_l_per_kwh"]), fac * float(w["ewif_excl_hydro_l_per_kwh"])
        prov(p, "off-site EWIF (hydropower allocated)", w, "ewif_l_per_kwh")
        prov(p, "off-site EWIF excluding hydropower", w, "ewif_excl_hydro_l_per_kwh")
        out.update(ml_offsite_reig=off * k, ml_offsite_excl_hydro=off_x * k,
                   ml_total_reig=(onsite + off) * k, ml_total_excl_hydro=(onsite + off_x) * k)
        warnings.append("two hydropower conventions are reported and neither is preferred: 'reig' allocates reservoir evaporation to electricity, 'excl_hydro' excludes it")
    return out
