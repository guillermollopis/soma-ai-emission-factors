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


def csv_value(name, **match):
    for r in rows(name):
        if all(r[k] == v for k, v in match.items()):
            return r
    raise AssertionError(match)


@pytest.mark.parametrize(
    "cls,key,region_id",
    [
        ("class_b_mid", "eu-west-1", "eu-west-1 (Ireland)"),
        ("class_a_small", "eu-north-1", "eu-north-1 (Sweden)"),
        ("class_c_large_moe", "ap-southeast-1", "ap-southeast-1 (Singapore)"),
    ],
)
def test_factor_matches_csv(cls, key, region_id):
    f = saef.factor(cls, key)
    elec = float(csv_value("carbon_factors.csv", model_class=cls, infra_scenario="H100-central", region_id=region_id)["kg_co2e_per_1m_tokens"])
    emb = float(csv_value("lifecycle_factors.csv", model_class=cls, term="embodied_hardware")["kg_co2e_per_1m_tokens"])
    trn = float(csv_value("lifecycle_factors.csv", model_class=cls, term="training_amortised")["kg_co2e_per_1m_tokens"])
    water = float(csv_value("water_factors.csv", model_class=cls, infra_scenario="H100-central", region_id=region_id)["total_ml_per_1k_tokens"])
    assert f["terms"]["electricity"] == elec
    assert f["kg_co2e_per_1m_tokens"] == pytest.approx(elec + emb + trn)
    assert f["water_ml_per_1m_tokens"] == pytest.approx(water * 1000)
    assert f["version"] == saef.__version__


def test_default_factor_and_prefix_region():
    assert saef.factor()["region"] == "us-east-1"
    assert saef.factor("B", "Ireland")["region"] == "eu-west-1"


def test_estimate_scales_linearly():
    a = saef.estimate(tokens=1_000_000, model_class="B", region="eu-west-1")
    b = saef.estimate(tokens=5_000_000, model_class="B", region="eu-west-1")
    assert b.kg_co2e == pytest.approx(5 * a.kg_co2e)
    assert b.water_ml == pytest.approx(5 * a.water_ml)
    assert a.kg_co2e == pytest.approx(saef.factor("B", "eu-west-1")["kg_co2e_per_1m_tokens"])
    assert a.sources["electricity"]["file"] == "carbon_factors.csv"
    assert a.sources["water"]["column"] == "total_ml_per_1k_tokens"


def test_estimate_by_model_and_reasoning_multiplier():
    plain = saef.estimate(tokens=1000, model="gpt-4o-mini", region="eu-west-1")
    assert plain.model_class == "class_a_small"
    assert plain.sources["model_class"]["file"] == "model_class_mapping.csv"
    reasoning = saef.estimate(tokens=1000, model="o3", region="eu-west-1")
    assert reasoning.reasoning_multiplier == 25
    assert reasoning.effective_tokens == 25000


@pytest.mark.parametrize(
    "model,cls",
    [
        ("gpt-4o", "class_b_mid"),
        ("gpt-4o-mini", "class_a_small"),
        ("gpt-4o-mini-2024-07-18", "class_a_small"),
        ("claude-sonnet-4-5", "class_b_mid"),
        ("anthropic/claude-haiku-4-5", "class_a_small"),
        ("gemini-2.5-flash", "class_a_small"),
    ],
)
def test_model_class_for(model, cls):
    assert saef.model_class_for(model) == cls


def test_unknown_region_raises_clear_error():
    with pytest.raises(saef.SaefError, match="Unknown region 'mars-1'.*Available regions"):
        saef.factor("B", "mars-1")
    with pytest.raises(ValueError):
        saef.estimate(tokens=10, model_class="B", region="atlantis")


def test_ambiguous_region_and_unknown_model():
    with pytest.raises(saef.SaefError, match="ambiguous"):
        saef.factor("B", "eu-west")
    with pytest.raises(saef.SaefError, match="not in model_class_mapping"):
        saef.model_class_for("totally-made-up-model")


def test_regions_and_citation():
    assert "eu-west-1" in saef.regions()
    assert "10.5281/zenodo.20443585" in saef.citation()


def test_agrees_with_check_factors_arithmetic():
    facility = float(csv_value("energy_factors.csv", model_class="class_b_mid", infra_scenario="H100 industry-avg (estimated)")["facility_wh_per_1k_tokens"])
    grid = float(csv_value("grid_intensity.csv", region_id="eu-west-1")["grid_kg_co2e_per_kwh"])
    assert saef.factor("B", "eu-west-1")["terms"]["electricity"] == pytest.approx(facility * grid, abs=6e-6)


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "saef", *args], capture_output=True, text=True)


def test_cli_estimate():
    p = run_cli("estimate", "--tokens", "1000000", "--model", "gpt-4o-mini", "--region", "eu-west-1")
    assert p.returncode == 0, p.stderr
    assert "kg CO2e:" in p.stdout
    assert "carbon_factors.csv" in p.stdout


def test_cli_error_exit_code():
    p = run_cli("estimate", "--tokens", "10", "--model-class", "B", "--region", "mars-1")
    assert p.returncode == 2
    assert "Unknown region" in p.stdout
