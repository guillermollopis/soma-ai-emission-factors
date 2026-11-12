import csv
import pathlib
import subprocess
import sys

import pytest

import saef

ROOT = pathlib.Path(__file__).resolve().parents[1]


def rows(name):
    with open(ROOT / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cloud_row(region_id):
    return next(r for r in rows("cloud_region_factors.csv") if r["region_id"] == region_id)


@pytest.mark.parametrize(
    "cls,region_id",
    [("B", "eu-west-1"), ("A", "eu-north-1"), ("C", "ap-southeast-1")],
)
def test_factor_matches_csv(cls, region_id):
    r = cloud_row(region_id)
    c = cls.lower()
    f = saef.factor(cls, region_id)
    assert f["kg_co2e_per_1m_billed_tokens"] == float(r[f"class_{c}_factor_kg_per_m_billed_tokens_default_mix"])
    assert f["kg_co2e_per_1m_effective_tokens"] == float(r[f"class_{c}_inference_footprint_kg_per_m_effective_tokens"])
    t = f["terms"]
    assert t["electricity"] + t["embodied"] + t["construction"] == pytest.approx(f["kg_co2e_per_1m_effective_tokens"], rel=1e-4)
    assert t["training_per_1m_billed_tokens"] == float(r[f"class_{c}_training_kg_per_m_billed_tokens"])
    assert f["version"] == saef.__version__ == "3.0.0"


def test_estimate_matches_hand_calculation():
    r = cloud_row("eu-west-1")
    e = saef.estimate(output_tokens=2e6, input_tokens=10e6, model_class="B", region="eu-west-1")
    t_eff = (2e6 + 0.3 * 10e6) / 1.339
    want = t_eff / 1e6 * float(r["class_b_inference_footprint_kg_per_m_effective_tokens"]) + 12 * float(
        r["class_b_training_kg_per_m_billed_tokens"]
    )
    assert e.kg_co2e == pytest.approx(want, rel=1e-4)
    assert e.kg_co2e == pytest.approx(sum(e.terms_kg_co2e.values()))


def test_estimate_scales_linearly_and_has_provenance():
    a = saef.estimate(tokens=1_000_000, model_class="B", region="eu-west-1")
    b = saef.estimate(tokens=5_000_000, model_class="B", region="eu-west-1")
    assert b.kg_co2e == pytest.approx(5 * a.kg_co2e)
    assert b.water_ml == pytest.approx(5 * a.water_ml)
    assert a.kg_co2e == pytest.approx(saef.factor("B", "eu-west-1")["kg_co2e_per_1m_billed_tokens"], rel=1e-4)
    assert any(s["file"] == "cloud_region_factors.csv" and s["line"] > 1 for s in a.provenance)
    assert any(s["file"] == "region_water_factors.csv" for s in a.provenance)
    assert a.kg_co2e_high_case > a.kg_co2e


def test_water_unavailable_where_no_factor():
    e = saef.estimate(tokens=1e6, model_class="A", region="DEU")
    assert e.water_ml is None
    assert any("water" in w for w in e.warnings)
    assert saef.estimate(tokens=1e6, model_class="A", region=None).water_ml is None


@pytest.mark.parametrize(
    "model,cls",
    [
        ("gpt-4o", "B"),
        ("gpt-4o-mini", "A"),
        ("gpt-4o-mini-2024-07-18", "A"),
        ("claude-sonnet-4-5", "B"),
        ("anthropic.claude-3-haiku-20240307-v1:0", "A"),
        ("gpt-5", "C"),
    ],
)
def test_model_class_for(model, cls):
    assert saef.model_class_for(model) == cls


def test_unmapped_model_defaults_to_class_c_with_warning():
    assert saef.model_class_for("totally-made-up-model") == "C"
    e = saef.estimate(tokens=1e6, model="totally-made-up-model", region="eu-west-1")
    assert e.model_class == "C"
    assert any("unmapped" in w for w in e.warnings)


def test_unknown_region_raises_clear_error():
    with pytest.raises(saef.SaefError, match="Unknown region 'mars-1'"):
        saef.factor("B", "mars-1")
    with pytest.raises(ValueError, match="Did you mean"):
        saef.estimate(tokens=10, model_class="B", region="us-eas")


def test_bad_inputs():
    with pytest.raises(saef.SaefError):
        saef.estimate(tokens=10)
    with pytest.raises(saef.SaefError):
        saef.estimate(tokens=10, model="gpt-4o", model_class="B")
    with pytest.raises(saef.SaefError):
        saef.estimate(tokens=-1, model_class="B")
    with pytest.raises(saef.SaefError):
        saef.factor("Z", "eu-west-1")


def test_coverage_counts_and_citation():
    assert len(saef.regions()) == 143
    assert len(saef.regions("aws")) == 37
    assert len(saef.grids()) == 240
    assert "eu-west-1" in saef.regions()
    assert "10.5281/zenodo.20443585" in saef.citation()


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "saef", *args], capture_output=True, text=True)


def test_cli_estimate():
    p = run_cli("estimate", "--tokens", "1000000", "--model", "gpt-4o-mini", "--region", "eu-west-1")
    assert p.returncode == 0, p.stderr
    assert "kg CO2e:" in p.stdout
    assert "cloud_region_factors.csv" in p.stdout


def test_cli_error_exit_code():
    p = run_cli("estimate", "--tokens", "10", "--model-class", "B", "--region", "mars-1")
    assert p.returncode == 2
    assert "Unknown region" in p.stdout


def test_parity_with_official_calculator():
    import importlib
    import os

    folder = os.environ.get("SAEF_OFFICIAL_DIR")
    if not folder:
        pytest.skip("set SAEF_OFFICIAL_DIR to the folder holding ai_inference_emissions.py")
    sys.path.insert(0, folder)
    try:
        off = importlib.import_module("ai_inference_emissions")
    finally:
        sys.path.remove(folder)
    for prov, reg in [(None, "eu-west-1"), ("azure", "swedencentral"), (None, "ap-southeast-1"), (None, "eu-south-2"), (None, "DEU"), (None, None)]:
        for model in ("gpt-4o", "gpt-4o-mini", "o3", "llama-3.1-70b"):
            o = off.emissions(model=model, provider=prov, region=reg, input_tokens=10e6, output_tokens=2e6)
            e = saef.estimate(model=model, provider=prov, region=reg, input_tokens=10e6, output_tokens=2e6)
            assert e.kg_co2e == pytest.approx(o["kg_co2e"], rel=1e-9)
            assert e.kg_co2e_high_case == pytest.approx(o["kg_co2e_high_case"], rel=1e-9)
            w = off.water(o["effective_tokens"], o["model_class"], prov, reg)
            if w["litres_total_reig"] is None:
                assert e.water_ml is None
            else:
                assert e.water_ml == pytest.approx(w["litres_total_reig"] * 1000, rel=1e-9)
