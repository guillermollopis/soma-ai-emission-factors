"""Recompute every H100-central carbon row (facility Wh x grid) and every water row (WUE, EWIF)
and fail if a published row differs by more than 0.000006 kg/1M tokens (carbon) or 0.0006 mL/1k tokens (water).
Facility Wh per 1k tokens is read from energy_factors.csv and checked against 0.04000 / 0.16167 / 0.20567.
Also cross-checks every H100-central carbon row, every water row, and both lifecycle terms against
the SOMA product's aiEmissionFactors.ts: path from the SOMA_PRODUCT_FACTORS environment variable, or
the default relative product location if that variable is unset; the parity check is skipped (not a
failure) if no product file is found at the resolved path.
Run: python tools/check_factors.py (from the repo root)"""
import csv, os, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUE = 1.20
CARBON_TOL = 0.000006
WATER_TOL = 0.0006
EXPECTED_FACILITY = {"class_a_small": 0.04000, "class_b_mid": 0.16167, "class_c_large_moe": 0.20567}

PRODUCT = pathlib.Path(os.environ.get("SOMA_PRODUCT_FACTORS") or (ROOT.parents[2] / "product" / "core" / "src" / "lib" / "aiEmissionFactors.ts"))
PARITY_TOL = 0.0006
LIFECYCLE_PARITY_TOL = 0.000006
REGION_MAP = {
    "us-east-1": "us_east", "us-west-2": "us_west", "us-south-central": "us_texas",
    "us-south": "us_south", "us-central1": "us_central", "eu-west-1": "eu_ireland",
    "eu-central-1": "eu_germany", "europe-west4": "eu_netherlands", "eu-north-1": "eu_sweden",
    "eu-west-3": "eu_france", "europe-north1": "eu_finland", "europe-west2": "uk_london",
    "europe-west1": "eu_belgium", "ap-northeast-1": "japan", "ap-southeast-1": "singapore",
    "global-average": "global_avg",
}
CLASS_MAP = {"class_a_small": "A", "class_b_mid": "B", "class_c_large_moe": "C"}

def parse_table(src, name):
    m = re.search(r"export const " + name + r"[^{]*\{(.*?)\n\};", src, re.S)
    if not m:
        die(f"{PRODUCT.name}: {name} not found")
    body = m.group(1)
    out = {}
    for cls, row in re.findall(r"\n\s*([ABC]):\s*\{([^}]*)\}", body):
        for key, val in re.findall(r"(\w+):\s*([0-9.]+)", row):
            out[(cls, key)] = float(val)
    return out

def parse_record(src, name):
    m = re.search(r"export const " + name + r"[^{]*\{([^}]*)\}", src)
    if not m:
        die(f"parity: could not find {name} in product module")
    return {k: float(v) for k, v in re.findall(r"(\w+):\s*([0-9.]+)", m.group(1))}

def read(name):
    with open(ROOT / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def region_key(s):
    return s.split(" (")[0].strip()

def die(msg):
    print(msg)
    sys.exit(1)

def load_facility():
    facility = {}
    for r in read("energy_factors.csv"):
        if "H100" not in r["infra_scenario"]:
            continue
        facility[r["model_class"]] = float(r["facility_wh_per_1k_tokens"])
    for cls, expect in EXPECTED_FACILITY.items():
        got = facility.get(cls)
        if got is None or abs(got - expect) > 1e-9:
            die(f"energy_factors.csv: {cls} facility_wh_per_1k_tokens {got} != expected {expect}")
    return facility

FACILITY = load_facility()
grid = {r["region_id"]: float(r["grid_kg_co2e_per_kwh"]) for r in read("grid_intensity.csv")}
bad = 0

for r in read("carbon_factors.csv"):
    if "H100" not in r["infra_scenario"]:
        continue
    region = region_key(r["region_id"])
    if region not in grid:
        die(f"carbon_factors.csv: unknown region {region}")
    if r["model_class"] not in FACILITY:
        die(f"carbon_factors.csv: unknown model_class {r['model_class']}")
    g = grid[region]
    fac = FACILITY[r["model_class"]]
    want = fac * g
    got = float(r["kg_co2e_per_1m_tokens"])
    if abs(want - got) > CARBON_TOL:
        bad += 1
        print(f"carbon {r['model_class']} {r['region_id']}: have {got}, expect {want:.5f}")
    if abs(float(r["grid_kg_co2e_per_kwh"]) - g) > 1e-9:
        bad += 1
        print(f"carbon {r['model_class']} {r['region_id']}: grid_kg_co2e_per_kwh {r['grid_kg_co2e_per_kwh']} != grid table {g}")
    if abs(float(r["facility_wh_per_1k_tokens"]) - fac) > 1e-9:
        bad += 1
        print(f"carbon {r['model_class']} {r['region_id']}: facility_wh_per_1k_tokens {r['facility_wh_per_1k_tokens']} != FACILITY {fac}")
    if abs(float(r["g_co2e_per_1k_tokens"]) - float(r["kg_co2e_per_1m_tokens"])) > 1e-9:
        bad += 1
        print(f"carbon {r['model_class']} {r['region_id']}: g_co2e_per_1k_tokens {r['g_co2e_per_1k_tokens']} != kg_co2e_per_1m_tokens {r['kg_co2e_per_1m_tokens']}")

for r in read("water_factors.csv"):
    if r["model_class"] not in FACILITY:
        die(f"water_factors.csv: unknown model_class {r['model_class']}")
    fac = FACILITY[r["model_class"]]
    s1 = fac / PUE * float(r["scope1_wue_l_per_kwh"])
    s2 = fac * float(r["scope2_ewif_l_per_kwh"])
    want = s1 + s2
    got = float(r["total_ml_per_1k_tokens"])
    if abs(want - got) > WATER_TOL:
        bad += 1
        print(f"water {r['model_class']} {r['region_id']}: have {got}, expect {want:.3f}")
    if abs(float(r["facility_wh_per_1k_tokens"]) - fac) > 1e-9:
        bad += 1
        print(f"water {r['model_class']} {r['region_id']}: facility_wh_per_1k_tokens {r['facility_wh_per_1k_tokens']} != FACILITY {fac}")
    if abs(float(r["pue"]) - PUE) > 1e-9:
        bad += 1
        print(f"water {r['model_class']} {r['region_id']}: pue {r['pue']} != {PUE}")

def check_parity(src):
    bad = 0
    token_factors = parse_table(src, "TOKEN_FACTORS")
    water_factors = parse_table(src, "WATER_FACTORS_ML_PER_1K")
    embodied = parse_record(src, "EMBODIED_KG_PER_MTOKEN")
    training = parse_record(src, "TRAINING_KG_PER_MTOKEN")
    tables = {"embodied_hardware": embodied, "training_amortised": training}

    found_carbon = set()
    for r in read("carbon_factors.csv"):
        if "H100" not in r["infra_scenario"]:
            continue
        pcls = CLASS_MAP.get(r["model_class"])
        pregion = REGION_MAP.get(region_key(r["region_id"]))
        if pcls is None or pregion is None:
            continue
        found_carbon.add((pcls, pregion))
        dataset = round(float(r["kg_co2e_per_1m_tokens"]), 3)
        product = token_factors.get((pcls, pregion))
        if product is None or abs(product - dataset) > PARITY_TOL:
            bad += 1
            print(f"product parity carbon {pcls} {pregion}: product {product}, dataset {dataset}")

    found_water = set()
    for r in read("water_factors.csv"):
        pcls = CLASS_MAP.get(r["model_class"])
        pregion = REGION_MAP.get(region_key(r["region_id"]))
        if pcls is None or pregion is None:
            continue
        found_water.add((pcls, pregion))
        dataset = round(float(r["total_ml_per_1k_tokens"]), 3)
        product = water_factors.get((pcls, pregion))
        if product is None or abs(product - dataset) > PARITY_TOL:
            bad += 1
            print(f"product parity water {pcls} {pregion}: product {product}, dataset {dataset}")

    for r in read("lifecycle_factors.csv"):
        pcls = CLASS_MAP.get(r["model_class"])
        if pcls is None:
            continue
        table = tables.get(r["term"])
        if table is None:
            die(f"lifecycle_factors.csv: unknown term {r['term']}")
        dataset = float(r["kg_co2e_per_1m_tokens"])
        product = table.get(pcls)
        if product is None or abs(product - dataset) > LIFECYCLE_PARITY_TOL:
            bad += 1
            print(f"product parity {r['term']} {pcls}: product {product}, dataset {dataset}")

    dataset_regions = {region_key(r["region_id"]) for r in read("carbon_factors.csv") if "H100" in r["infra_scenario"]}
    for region in sorted(dataset_regions):
        if region not in REGION_MAP:
            bad += 1
            print(f"product parity: dataset region {region} has no product key")

    for cls, key in sorted(token_factors):
        if (cls, key) not in found_carbon:
            bad += 1
            print(f"product parity: product key {cls} {key} has no dataset row")

    for cls, key in sorted(water_factors):
        if (cls, key) not in found_water:
            bad += 1
            print(f"product parity: product key {cls} {key} has no dataset row")

    return bad

if not PRODUCT.exists():
    print(f"parity: product module not found at {PRODUCT}, skipped")
else:
    bad += check_parity(PRODUCT.read_text(encoding="utf-8"))

print("OK" if bad == 0 else f"{bad} mismatches")
sys.exit(1 if bad else 0)
