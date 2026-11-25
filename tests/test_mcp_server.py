import pytest

pytest.importorskip("mcp")

import saef
from saef import mcp_server as m


def test_estimate_footprint_matches_core():
    r = m.estimate_footprint(1_000_000, "gpt-4o", "eu-west-1")
    e = saef.estimate(tokens=1_000_000, model="gpt-4o", region="eu-west-1")
    assert r["carbon"]["kg_co2e"] == e.kg_co2e
    assert r["carbon"]["g_co2e"] == pytest.approx(e.kg_co2e * 1000)
    assert r["carbon"]["high_case_kg_co2e"] >= r["carbon"]["kg_co2e"]
    assert r["water"]["ml"] == e.water_ml
    assert r["model_class"] == "B"
    assert r["region_used"] == "eu-west-1"
    assert r["dataset_version"] == saef.DATASET_VERSION
    assert r["provenance"]
    assert "not a measurement" in r["note"]


def test_energy_matches_readme_figure_for_class_b_chat():
    r = m.estimate_chat_usage(1, "gpt-4o")
    assert r["energy"]["facility_wh"] == pytest.approx(0.099, abs=0.001)
    assert "1210" in r["assumption"]


def test_chat_usage_scales_linearly():
    one = m.estimate_chat_usage(1, "gpt-4o", "eu-west-1")
    ten = m.estimate_chat_usage(10, "gpt-4o", "eu-west-1")
    assert ten["carbon"]["kg_co2e"] == pytest.approx(one["carbon"]["kg_co2e"] * 10)
    assert ten["messages"] == 10


def test_global_region_has_no_water():
    r = m.estimate_footprint(1000, "gpt-4o")
    assert r["water"]["ml"] is None
    assert r["region_used"] == "global-average"


def test_model_class():
    r = m.model_class("gpt-4o")
    assert r["model_class"] == "B"
    assert r["description"]
    assert m.model_class("totally-unknown-model")["model_class"] == "C"


def test_out_of_scope_model_raises():
    with pytest.raises(saef.SaefError):
        m.model_class("text-embedding-3-small")


def test_estimate_modality():
    r = m.estimate_modality(1_000_000, "input_tokens", "text-embedding-3-small", "eu-west-1")
    assert r["model_type"] == "embedding_text" and r["dataset_version"] == saef.DATASET_VERSION
    assert r["carbon"]["low_kg_co2e"] < r["carbon"]["kg_co2e"] < r["carbon"]["high_kg_co2e"]
    assert r["energy"]["facility_wh"] > 0 and r["water"]["ml"] > 0
    with pytest.raises(ValueError):
        m.estimate_modality(-1, "images", "dall-e-3")
    with pytest.raises(saef.SaefError):
        m.estimate_modality(1, "images", "gpt-4o-mini-tts")


def test_list_regions():
    r = m.list_regions()
    ids = {c["region"] for c in r["cloud_regions"]}
    assert "eu-west-1" in ids
    assert "ESP" in r["country_codes"]
    assert "global" in r["special"]


def test_compare_regions_sorted_with_bad_region_reported():
    r = m.compare_regions(1_000_000, "gpt-4o", ["us-east-1", "eu-north-1", "nowhere-land"])
    names = [x["region"] for x in r["regions"]]
    assert set(names) == {"us-east-1", "eu-north-1"}
    assert r["regions"][0]["kg_co2e"] <= r["regions"][1]["kg_co2e"]
    assert names[0] == "eu-north-1"
    assert r["errors"][0]["region"] == "nowhere-land"
    assert r["model_class"] == "B"


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        m.estimate_footprint(-1, "gpt-4o")
    with pytest.raises(ValueError):
        m.estimate_chat_usage(-1, "gpt-4o")


def test_about_resource_and_registration():
    text = m.about()
    for s in ("10.5281/zenodo.20443585", "2606.10660", "https://somaai.earth/ai-footprint", "Non-Commercial", "https://somaai.earth/licensing"):
        assert s in text


def test_tools_and_resource_registered():
    import asyncio

    tools = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert tools == {"estimate_footprint", "estimate_chat_usage", "estimate_modality", "model_class", "list_regions", "compare_regions"}
    uris = {str(r.uri) for r in asyncio.run(m.mcp.list_resources())}
    assert "saef://about" in uris
