import csv
import json
import pathlib
import subprocess
import sys

import pytest

import saef
from saef import _calc
from saef.integrations._base import Tracker
from saef.integrations.openai import estimate_from_response

ROOT = pathlib.Path(__file__).resolve().parents[1]


def rows(name):
    with open(ROOT / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


FACTORS = {(r["model_type"], r["class"], r["functional_unit"]): r for r in rows("modality_factors.csv")}


def server(model_type, cls, unit, case="central"):
    return float(FACTORS[model_type, cls, unit][f"server_wh_{case}"])


def test_generator_reproduces_the_committed_tables(tmp_path):
    work = tmp_path / "repo"
    (work / "scripts").mkdir(parents=True)
    for p in ROOT.glob("*.csv"):
        (work / p.name).write_bytes(p.read_bytes())
    (work / "modality_factors.csv").unlink()
    (work / "scripts/build_modality_factors.py").write_bytes((ROOT / "scripts/build_modality_factors.py").read_bytes())
    subprocess.run([sys.executable, "-I", str(work / "scripts/build_modality_factors.py")], check=True, capture_output=True)
    assert (work / "modality_factors.csv").read_text() == (ROOT / "modality_factors.csv").read_text()
    for name in ("modality_factors.csv", "modality_sources.csv", "model_class_mapping.csv"):
        assert (ROOT / name).read_bytes() == (ROOT / "src/saef/data" / name).read_bytes()


# Server-level release 1 values (energy_per_unit.csv), which the GPU-level rows must give back through H x I.
R1_SERVER = [
    ("embedding_text", "E-small", "1k_input_tokens", 0.000477, 7.52e-05, 0.0434),
    ("embedding_text", "E-large", "1k_input_tokens", 0.0134, 0.00157, 0.376),
    ("embedding_image", "E-small", "image", 0.000123, 1.93e-05, 0.00192),
    ("embedding_image", "E-large", "image", 0.0134, 0.000402, 0.376),
    ("rerank", "R-small", "1k_input_tokens", 0.000798, 0.000126, 0.00965),
    ("rerank", "R-small", "search_unit", 0.0239, 0.000629, 0.483),
    ("rerank", "R-large", "1k_input_tokens", 0.0134, 0.00157, 0.376),
    ("moderation", "M-B", "1k_input_tokens", 0.044, 0.0134, 0.0544),
    ("moderation", "M-A", "1k_input_tokens", 0.0134, 0.00803, 0.0198),
    ("moderation", "M-B", "call", 0.022, 0.000282, 0.136),
    ("image_generation_diffusion", "D-small", "image", 0.446, 0.191, 0.725),
    ("image_generation_diffusion", "D-large", "image", 2.68, 2.03, 3.64),
    ("speech_to_text", "STT-ENC", "audio_minute", 0.0497, 0.00862, 0.286),
    ("speech_to_text", "STT-LLM-A", "audio_minute", 0.0224, 0.0102, 0.0448),
    ("speech_to_text", "STT-LLM-B", "audio_minute", 0.0733, 0.0344, 0.123),
]


@pytest.mark.parametrize("mt,cls,unit,c,lo,hi", R1_SERVER)
def test_gpu_rows_give_back_the_server_level_inputs(mt, cls, unit, c, lo, hi):
    r = FACTORS[mt, cls, unit]
    for case, want in (("central", c), ("low", lo), ("high", hi)):
        assert float(r[f"gpu_wh_{case}"]) * 1.5 * 1.1 == pytest.approx(want, rel=5e-3)


def test_chain_is_the_text_chain():
    text = next(r for r in rows("factors_v3_lifecycle.csv") if r["model_class"] == "A" and r["gpu"] == "H100")
    a = next(r for r in rows("cloud_region_factors.csv") if r["region_id"] == "eu-west-1")
    for r in FACTORS.values():
        gpu, srv, fac = (float(r[f"{k}_wh_central"]) for k in ("gpu", "server", "facility"))
        assert srv == pytest.approx(gpu * float(text["h_host_uplift"]) * float(text["i_idle_uplift"]), rel=1e-5)
        assert fac == pytest.approx(srv * float(text["pue"]), rel=1e-5)
        per_kwh = float(a["class_a_embodied_kg_per_m_effective_tokens"]) / 0.0599777
        assert float(r["embodied_g_co2e_central"]) == pytest.approx(srv * per_kwh, rel=1e-4)
        assert float(r["construction_g_co2e_central"]) == pytest.approx(srv * 0.00766, rel=1e-4)
        assert float(r["facility_wh_high_case"]) == pytest.approx(gpu * 1.88 * 1.2 * 1.45, rel=1e-5)
        assert float(r["gpu_wh_low"]) <= gpu <= float(r["gpu_wh_high"])


@pytest.mark.parametrize("cls", ["A", "B"])
def test_token_rows_match_the_text_calculator(cls):
    e = saef.estimate_modality(model_type="speech_to_text", model_class=f"STT-LLM-{cls}", unit_count=1e6, unit="text_output_tokens", region="eu-west-1")
    t = saef.estimate(output_tokens=1e6, model_class=cls, region="eu-west-1")
    assert e.terms_kg_co2e["electricity"] == pytest.approx(t.terms_kg_co2e["electricity"], rel=1e-3)
    assert e.terms_kg_co2e["construction"] == pytest.approx(t.terms_kg_co2e["construction"], rel=1e-3)
    assert e.water_ml == pytest.approx(t.water_ml, rel=1e-3)
    if cls == "A":
        assert e.terms_kg_co2e["embodied"] == pytest.approx(t.terms_kg_co2e["embodied"], rel=1e-3)


def test_reference_region_columns_match_the_calculator():
    units = {f: u for u, (f, _s) in saef._modality.UNITS.items()}
    for (mt, cls, fu), r in FACTORS.items():
        if fu not in units or fu in ("1k_image_output_tokens", "1k_image_output_tokens_parity"):
            continue
        for rid in ("eu-west-1", "us-east-1", "global-average"):
            col = rid.replace("-", "_")
            e = saef.estimate_modality(model_type=mt, model_class=cls, unit_count=1, unit=fu, region=rid)
            for case, got in (("central", e.kg_co2e), ("low", e.kg_co2e_low), ("high", e.kg_co2e_high), ("high_case", e.kg_co2e_high_case)):
                assert got * 1000 == pytest.approx(float(r[f"co2e_g_{col}_{case}"]), rel=1e-4)
            if rid == "global-average":
                assert e.water_ml is None
            else:
                assert e.water_ml == pytest.approx(float(r[f"water_ml_{col}_central"]), rel=1e-4)
                assert e.water_ml_high == pytest.approx(float(r[f"water_ml_{col}_high"]), rel=1e-4)
                assert e.water_ml_excl_hydro == pytest.approx(float(r[f"water_ml_excl_hydro_{col}_central"]), rel=1e-4)


CUSTOMER = [
    ("gpt-image-2", "image_generation_token_billed", "IMG-TOK-B"),
    ("gpt-image-2-tpp-rgba", "image_generation_token_billed", "IMG-TOK-B"),
    ("img-1p-p-ev3", "text", "C"),
    ("gpt-image-1.5-2025-12-16", "image_generation_token_billed", "IMG-TOK-B"),
    ("gpt-image-2.5-flare", "image_generation_token_billed", "IMG-TOK-B"),
    ("gpt-image-2.5-sunburst", "image_generation_token_billed", "IMG-TOK-B"),
    ("gemini-3.1-flash-lite-image", "image_generation_token_billed", "IMG-TOK-A"),
    ("advanced-voice-4o", "realtime_turn_based", "RT-B"),
    ("advanced-voice-bidi-instant", "realtime_full_duplex", "FD-A"),
    ("gpt-4o-mini-transcribe-2025-12-15", "speech_to_text", "STT-LLM-A"),
    ("gpt-4o-mini-tts", "tts", "TTS-B"),
    ("gpt-live-1.1-codex", "realtime_full_duplex", "FD-A"),
    ("TitanEmbeddingV2-Text", "embedding_text", "E-small"),
    ("TitanEmbeddingsG1-Image", "embedding_text;embedding_image", "E-large"),
]


@pytest.mark.parametrize("model,mt,cls", CUSTOMER)
def test_customer_names(model, mt, cls):
    assert saef.model_type_for(model) == (mt, cls)


def test_img_1p_stays_unmapped_and_flagged():
    assert _calc.map_model("img-1p-p-ev3")["is_default_fallback"]
    e = saef.estimate_modality("img-1p-p-ev3", 3, "images", "eu-west-1")
    assert (e.model_type, e.model_class, e.class_basis) == ("image_generation_diffusion", "D-large", "billing unit")
    assert any("unmapped" in w for w in e.warnings) and any("routed by the billing unit" in w for w in e.warnings)


@pytest.mark.parametrize(
    "model,mt,cls",
    [
        ("text-embedding-3-small", "embedding_text", "E-small"),
        ("text-embedding-3-large", "embedding_text", "E-large"),
        ("text-embedding-004", "embedding_text", "E-large"),
        ("amazon.titan-embed-text-v2:0", "embedding_text", "E-small"),
        ("TitanEmbeddingsG1-Text", "embedding_text", "E-large"),
        ("amazon.titan-embed-image-v1", "embedding_text;embedding_image", "E-large"),
        ("cohere.embed-v4:0", "embedding_text;embedding_image", "E-large"),
        ("cohere.embed-english-light-v3", "embedding_text", "E-small"),
        ("cohere.rerank-v3-5:0", "rerank", "R-small"),
        ("BAAI/bge-reranker-v2-m3", "rerank", "R-small"),
        ("BAAI/bge-m3", "embedding_text", "E-small"),
        ("Qwen/Qwen3-Embedding-8B", "embedding_text", "E-large"),
        ("omni-moderation-latest", "moderation", "M-B"),
        ("text-moderation-007", "moderation", "M-A"),
        ("gpt-image-1-mini", "image_generation_token_billed", "IMG-TOK-A"),
        ("gemini-3-pro-image-preview", "image_generation_token_billed", "IMG-TOK-B"),
        ("dall-e-3", "image_generation_diffusion", "D-large"),
        ("dall-e-2", "image_generation_diffusion", "D-small"),
        ("black-forest-labs/FLUX.1-schnell", "image_generation_diffusion", "D-small"),
        ("black-forest-labs/FLUX.1-dev", "image_generation_diffusion", "D-large"),
        ("stability.sd3-5-large-v1:0", "image_generation_diffusion", "D-large"),
        ("whisper-1", "speech_to_text", "STT-ENC"),
        ("gpt-4o-transcribe-diarize", "speech_to_text", "STT-LLM-B"),
        ("chirp-3-hd", "tts", "TTS-B"),
        ("gemini-2.5-flash-preview-tts", "tts", "TTS-B"),
        ("sora-2", "video_generation", "V-A"),
        ("veo-3.1-fast-generate-001", "video_generation", "V-A"),
        ("veo-3.1-generate-001", "video_generation", "V-B"),
        ("sora-2-pro", "video_generation", "V-C"),
        ("kling-v2.1-pro", "video_generation", "V-B"),
        ("kling-v2-master", "video_generation", "V-C"),
        ("Wan-AI/Wan2.1-T2V-1.3B", "video_generation", "V-A"),
        ("Wan-AI/Wan2.1-T2V-14B", "video_generation", "V-B"),
        ("amazon.nova-reel-v1:0", "video_generation", "V-A"),
        ("gen4_turbo", "video_generation", "V-A"),
        ("gen4.5", "video_generation", "V-B"),
        ("luma.ray-v2:0", "video_generation", "V-B"),
        ("gemini-omni-flash-video", "video_generation", "V-A"),
        ("gemini-omni-flash", "text", "B"),
        ("gpt-realtime-mini-2025-10-06", "realtime_turn_based", "RT-A"),
        ("gpt-4o-realtime-preview", "realtime_turn_based", "RT-B"),
        ("gpt-4o-mini-realtime-preview", "realtime_turn_based", "RT-A"),
        ("gemini-2.5-flash-native-audio", "realtime_turn_based", "RT-A"),
        ("gemini-2.5-flash-live", "realtime_turn_based", "RT-A"),
        ("amazon.nova-2-sonic-v1:0", "realtime_turn_based", "RT-A"),
        ("gpt-realtime-whisper", "speech_to_text", "STT-ENC"),
        ("gpt-5-mini-tts", "tts", "TTS-B"),
        ("tts-1-hd", "tts", "TTS-B"),
        ("gpt-4o-audio-preview", "out-of-scope", None),
        ("luminous-base", "text", "C"),
        ("llama-guard-3-8b", "text", "A"),
        ("gemini-omni", "text", "B"),
        ("gpt-4o", "text", "B"),
    ],
)
def test_model_types(model, mt, cls):
    assert saef.model_type_for(model) == (mt, cls)


R2_GPU = [
    ("tts", "TTS-B", "audio_minute", 0.03768, 0.02056, 0.206),
    ("tts", "TTS-B", "1k_characters", 0.04186, 0.01958, 0.2747),
    ("realtime_turn_based", "RT-A", "1k_audio_output_tokens", 0.02987, 0.01622, 0.05996),
    ("realtime_turn_based", "RT-B", "user_audio_minute", 0.016, 0.003289, 0.02639),
    ("realtime_turn_based", "RT-B", "assistant_audio_minute", 0.1347, 0.07646, 0.2194),
    ("realtime_turn_based", "RT-B", "context_minute", 0.1127, 0.005572, 1.634),
    ("realtime_turn_based", "RT-B", "conversation_minute", 0.1797, 0.03332, 1.768),
    ("realtime_turn_based", "RT-C", "conversation_minute", 0.6692, 0.06676, 8.245),
    ("realtime_full_duplex", "FD-A", "active_session_minute", 0.04236, 0.01338, 0.09594),
    ("video_generation", "V-A", "video_second_720p", 11.93, 2.426, 38.97),
    ("video_generation", "V-B", "video_second_720p", 41.68, 10.94, 170.5),
    ("video_generation", "V-B", "video_second_1080p", 93.79, 13.13, 733.3),
    ("video_generation", "V-C", "video_second_2160p", 312.6, 83.37, 4263),
]


@pytest.mark.parametrize("mt,cls,unit,c,lo,hi", R2_GPU)
def test_release_2_rows_are_the_gpu_level_inputs(mt, cls, unit, c, lo, hi):
    r = FACTORS[mt, cls, unit]
    for case, want in (("central", c), ("low", lo), ("high", hi)):
        assert float(r[f"gpu_wh_{case}"]) == pytest.approx(want, rel=1e-3)


def test_eight_second_class_b_720p_clip_is_hundreds_of_wh():
    e = saef.estimate_modality("veo-3.1-generate-001", 8, "video_seconds", "us-east-1")
    assert e.model_class == "V-B" and 200 < e.gpu_energy_wh < 500 and 400 < e.energy_wh < 900
    assert saef.estimate_modality("veo-3.1-generate-001", 1, "video_clips", "us-east-1").energy_wh == pytest.approx(e.energy_wh, rel=1e-4)
    hd = saef.estimate_modality("veo-3.1-generate-001", 8, "video_seconds_1080p", "us-east-1")
    assert hd.energy_wh == pytest.approx(2.25 * e.energy_wh, rel=1e-4)


def test_gemini_omni_routes_by_billing_unit():
    v = saef.estimate_modality("gemini-omni-flash", 8, "video_seconds")
    assert (v.model_type, v.model_class) == ("video_generation", "V-A") and any("routed by the billing unit" in w for w in v.warnings)
    u = saef.estimate_modality("gemini-omni", 8, "video_seconds")
    assert (u.model_type, u.model_class) == ("video_generation", "V-C")
    t = saef.estimate(model="gemini-omni-flash", input_tokens=1000, output_tokens=100)
    assert t.model_class == "B" and not isinstance(t, saef.ModalityEstimate)
    assert saef.estimate_modality("grok-imagine", 1, "images").model_type == "image_generation_diffusion"


def test_voice_token_path_is_preferred_and_minutes_are_the_fallback():
    usage = {"input_tokens": 9000, "cached_input_tokens": 8000, "text_output_tokens": 400, "audio_output_tokens": 2700}
    e = saef.estimate(model="gpt-realtime", usage=usage, region="eu-west-1")
    assert (e.model_type, e.model_class) == ("realtime_turn_based", "RT-B") and e.terms_kg_co2e["training"] == 0
    t = saef.estimate(model_class="B", input_tokens=9000, cached_input_tokens=8000, output_tokens=400 + 1.1 * 2700, region="eu-west-1")
    assert e.terms_kg_co2e["electricity"] == pytest.approx(t.terms_kg_co2e["electricity"], rel=1e-3)
    plain = saef.estimate(model="gpt-realtime", input_tokens=10, output_tokens=10)
    assert plain.usage["audio_output_tokens"] == 10
    assert saef.estimate_modality("advanced-voice-4o", 10, "conversation_minutes").model_class == "RT-B"
    assert saef.estimate_modality("advanced-voice-bidi-instant", 10, "active_session_minutes").model_class == "FD-A"
    with pytest.raises(saef.SaefError, match="do not combine"):
        saef.estimate(model="gpt-realtime", usage={"conversation_minutes": 5, "audio_output_tokens": 100})
    with pytest.raises(saef.SaefError, match="active_session_minutes"):
        saef.estimate(model="gpt-live-1.1-codex", input_tokens=10, output_tokens=10)


def test_speech_units():
    minute = saef.estimate_modality("gpt-4o-mini-tts", 1, "audio_minutes")
    assert saef.estimate_modality("tts-1", 900, "characters").energy_wh == pytest.approx(minute.energy_wh, rel=1e-3)
    tokens = saef.estimate(model="gpt-4o-mini-tts", usage={"input_tokens": 225, "audio_output_tokens": 1200})
    assert tokens.energy_wh == pytest.approx(minute.energy_wh, rel=1e-3)
    assert saef.estimate_modality("gpt-4o-mini-tts", 60, "audio_seconds").model_type == "tts"


def test_text_class_of_a_modality_model_is_an_error():
    with pytest.raises(saef.SaefError, match="estimate_modality"):
        saef.model_class_for("text-embedding-3-small")


def test_gpt_image_high_quality_sits_inside_the_open_diffusion_range():
    lo = server("image_generation_diffusion", "D-small", "image", "low")
    hi = server("image_generation_diffusion", "D-large", "image", "high")
    by_row = saef.estimate_modality("gpt-image-1.5", 1, "images_1024_high")
    by_tokens = saef.estimate(model="gpt-image-1.5", usage={"image_output_tokens": 4160, "images": 1})
    assert by_row.server_energy_wh == pytest.approx(by_tokens.server_energy_wh, rel=1e-4)
    assert lo < by_row.server_energy_wh < hi
    assert lo < saef.estimate(model="gpt-image-1.5", usage={"image_output_tokens": 4160}).server_energy_wh < hi


def test_closed_image_models_never_fall_below_open_diffusion():
    dsmall = server("image_generation_diffusion", "D-small", "image")
    smallest = server("image_generation_diffusion", "D-small", "image", "low")
    for model, usage in [
        ("gpt-image-2", {"image_output_tokens": 272, "images": 1}),
        ("gpt-image-2", {"image_output_tokens": 272}),
        ("gpt-image-1-mini", {"images_1024_low": 1}),
        ("gemini-3.1-flash-lite-image", {"image_output_tokens": 1120, "images": 1}),
        ("gemini-3.1-flash-lite-image", {"images": 1}),
        ("gemini-3-pro-image", {"images_4k": 1}),
    ]:
        e = saef.estimate(model=model, usage=usage, region="eu-west-1")
        assert e.server_energy_wh >= dsmall * 0.9999, (model, usage)
        assert e.energy_wh_low / 1.2 >= smallest * 0.9999
        assert e.kg_co2e_low <= e.kg_co2e <= e.kg_co2e_high
    for (mt, cls, fu), r in FACTORS.items():
        if mt == "image_generation_token_billed" and fu.startswith("image"):
            assert float(r["gpu_wh_low"]) * 1.65 >= smallest * 0.9999
            assert float(r["gpu_wh_central"]) * 1.65 >= dsmall * 0.9999


def test_gpt_image_high_bound_reaches_class_c():
    for fu in ("1k_image_output_tokens", "1k_image_output_tokens_parity", "image_1024_low", "image_1024_medium", "image_1024_high", "image"):
        b, c = (FACTORS["image_generation_token_billed", f"IMG-TOK-{x}", fu] for x in "BC")
        parity = float(FACTORS["image_generation_token_billed", "IMG-TOK-C", "1k_image_output_tokens_parity"]["gpu_wh_central"])
        n = {"1k_image_output_tokens": 1, "1k_image_output_tokens_parity": 1, "image_1024_low": 0.272, "image_1024_medium": 1.056, "image_1024_high": 4.16, "image": 4.16}[fu]
        assert float(b["gpu_wh_high"]) >= parity * n * 0.9999


def test_image_output_token_is_one_output_token_of_the_text_class():
    parity = float(FACTORS["image_generation_token_billed", "IMG-TOK-B", "1k_image_output_tokens_parity"]["gpu_wh_central"])
    assert parity == pytest.approx(0.119037 / 1.339, rel=1e-4)


def test_image_count_floors_aggregated_tokens():
    parity_heavy = saef.estimate(model="gpt-image-2", usage={"image_output_tokens": 4160 * 100, "images": 100})
    floor_heavy = saef.estimate(model="gpt-image-2", usage={"image_output_tokens": 272 * 100, "images": 100})
    assert floor_heavy.server_energy_wh == pytest.approx(100 * server("image_generation_diffusion", "D-small", "image"), rel=1e-4)
    assert floor_heavy.components[0]["floored_to_diffusion"]
    assert parity_heavy.server_energy_wh > floor_heavy.server_energy_wh
    with pytest.raises(saef.SaefError, match="twice"):
        saef.estimate(model="gpt-image-2", usage={"image_output_tokens": 272, "images_1024_low": 1})
    assert saef.estimate(model="gpt-image-2", usage={"image_output_tokens": 0}).kg_co2e == 0


def test_every_model_type_estimates_with_low_central_high():
    for model, count, unit in [
        ("text-embedding-3-small", 1e6, "input_tokens"), ("TitanEmbeddingsG1-Image", 10, "images"),
        ("TitanEmbeddingsG1-Image", 1e6, "input_tokens"), ("cohere.rerank-v3-5:0", 100, "search_units"),
        ("rerank-v3.5", 1e6, "tokens_scored"), ("omni-moderation-latest", 100, "calls"),
        ("gpt-image-2", 1e5, "image_output_tokens"), ("dall-e-3", 4, "images"), ("whisper-1", 60, "audio_minutes"),
        ("gpt-4o-mini-transcribe-2025-12-15", 60, "audio_minutes"),
        ("gpt-4o-mini-tts", 1e5, "audio_output_tokens"), ("tts-1", 1e6, "characters"), ("gpt-realtime", 1e6, "audio_output_tokens"),
        ("advanced-voice-4o", 60, "conversation_minutes"), ("gpt-live-1.1-codex", 60, "active_session_minutes"),
        ("sora-2", 80, "video_seconds"), ("veo-3.1-generate-001", 8, "video_seconds_2160p"),
    ]:
        e = saef.estimate_modality(model, count, unit, "us-east-1")
        assert 0 < e.kg_co2e_low < e.kg_co2e < e.kg_co2e_high and e.kg_co2e < e.kg_co2e_high_case
        assert 0 < e.energy_wh_low < e.energy_wh < e.energy_wh_high
        assert 0 < e.water_ml_low < e.water_ml < e.water_ml_high and e.water_ml_excl_hydro < e.water_ml
        assert sum(e.terms_kg_co2e.values()) == pytest.approx(e.kg_co2e) and e.terms_kg_co2e["training"] == 0
        assert e.version == saef.DATASET_VERSION
        assert any(s["file"] == "modality_factors.csv" and s["line"] > 1 for s in e.provenance)
        json.dumps(e.to_dict())


def test_units_scale_and_alias():
    a = saef.estimate_modality("whisper-1", 2, "audio_minutes")
    assert saef.estimate_modality("whisper-1", 120, "audio_seconds").kg_co2e == pytest.approx(a.kg_co2e)
    assert saef.estimate_modality("whisper-1", 2, "audio_minute").kg_co2e == pytest.approx(a.kg_co2e)
    t = saef.estimate_modality("text-embedding-3-small", 5000, "input_tokens")
    assert saef.estimate_modality("text-embedding-3-small", 5, "1k_input_tokens").kg_co2e == pytest.approx(t.kg_co2e)
    big = saef.estimate_modality("dall-e-3", 1, "images", megapixels=1.835008)
    assert big.kg_co2e == pytest.approx(1.75 * saef.estimate_modality("dall-e-3", 1, "images").kg_co2e)
    small = saef.estimate_modality("dall-e-3", 1, "images", megapixels=0.25)
    assert small.kg_co2e == pytest.approx(saef.estimate_modality("dall-e-3", 1, "images").kg_co2e)


def test_main_estimate_routes_usage_records():
    e = saef.estimate(tokens=1e6, model="text-embedding-3-small", region="eu-west-1")
    assert isinstance(e, saef.ModalityEstimate)
    assert e.kg_co2e == pytest.approx(saef.estimate_modality("text-embedding-3-small", 1e6, "input_tokens", "eu-west-1").kg_co2e)
    assert saef.estimate(model="whisper-1", usage={"audio_minutes": 3}).model_type == "speech_to_text"
    g = saef.estimate(model="gpt-image-2", input_tokens=100, output_tokens=4160)
    assert g.usage == {"input_tokens": 100.0, "image_output_tokens": 4160.0}
    s = saef.estimate(model="gpt-4o-mini-transcribe", input_tokens=1000, output_tokens=200)
    m = saef.estimate_modality("gpt-4o-mini-transcribe", 1, "audio_minutes")
    assert s.kg_co2e == pytest.approx(m.kg_co2e, rel=1e-4)
    text = saef.estimate(model="gpt-4o", usage={"input_tokens": 1000, "output_tokens": 100}, region="eu-west-1")
    assert text.kg_co2e == saef.estimate(model="gpt-4o", input_tokens=1000, output_tokens=100, region="eu-west-1").kg_co2e


def test_billing_unit_decides_before_the_name():
    e = saef.estimate(model="gpt-5", usage={"image_output_tokens": 1000})
    assert (e.model_type, e.model_class) == ("image_generation_token_billed", "IMG-TOK-C")
    assert any("routed by the billing unit" in w for w in e.warnings)
    assert saef.estimate_modality("gpt-4o", 1, "audio_minutes").model_class == "STT-LLM-B"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(model="whisper-1", unit_count=5, unit="input_tokens"), "not charged on input_tokens"),
        (dict(model="gpt-4o", unit_count=5, unit="input_tokens"), "text factors"),
        (dict(model="dall-e-3", unit_count=5, unit="parsecs"), "unknown usage unit"),
        (dict(model="dall-e-3", unit_count=-1, unit="images"), ">= 0"),
        (dict(model="dall-e-3", unit_count=float("nan"), unit="images"), ">= 0"),
        (dict(model="lyria-2", unit_count=1, unit="images"), "out of scope"),
        (dict(model="gpt-4o-audio-preview", unit_count=1, unit="audio_minutes"), "out of scope"),
        (dict(model_type="rerank", model_class="Z", unit_count=1, unit="search_units"), "no modality factors"),
        (dict(model="dall-e-3", unit_count=1, unit="images", region="mars-1"), "Unknown region"),
    ],
)
def test_modality_errors(kwargs, match):
    with pytest.raises(saef.SaefError, match=match):
        saef.estimate_modality(**kwargs)


def test_token_arguments_that_do_not_fit_are_errors():
    with pytest.raises(saef.SaefError, match="output tokens"):
        saef.estimate(model="text-embedding-3-small", output_tokens=5, input_tokens=5)
    with pytest.raises(saef.SaefError, match="split"):
        saef.estimate(model="gpt-image-2", tokens=5000)
    with pytest.raises(saef.SaefError, match="audio_minutes"):
        saef.estimate(model="whisper-1", tokens=5000)
    with pytest.raises(saef.SaefError, match="not both"):
        saef.estimate(model="gpt-4o", tokens=5, usage={"input_tokens": 5})


def test_usage_tokens_is_a_billed_total():
    assert saef.estimate(model="gpt-4o", usage={"tokens": 1e6}).kg_co2e == saef.estimate(model="gpt-4o", tokens=1e6).kg_co2e
    assert saef.estimate(model="text-embedding-3-small", usage={"tokens": 1e6}).model_type == "embedding_text"
    with pytest.raises(saef.SaefError, match="split"):
        saef.estimate(model="gpt-image-2", usage={"tokens": 5000})
    with pytest.raises(saef.SaefError, match="combined"):
        saef.estimate(model="gpt-4o", usage={"tokens": 5000, "output_tokens": 5})
    with pytest.raises(saef.SaefError, match="unknown usage unit"):
        saef.estimate_modality("gpt-image-2", 5000, "tokens")


def test_tracker_and_response_objects():
    t = Tracker(region="eu-west-1")
    assert t.record("text-embedding-3-small", 1000, 0)["model_class"] == "E-small"
    assert t.record("whisper-1", 0, 0) is None
    img = {"usage": {"input_tokens": 50, "output_tokens": 544, "input_tokens_details": {"text_tokens": 50, "image_tokens": 0}}, "data": [{}, {}]}
    e = estimate_from_response(img, region="eu-west-1", model="gpt-image-1.5")
    assert e.usage["images"] == 2 and e.components[-1]["floored_to_diffusion"]
    emb = {"model": "text-embedding-3-small", "usage": {"prompt_tokens": 8, "total_tokens": 8}, "data": [{}]}
    assert estimate_from_response(emb).model_type == "embedding_text"
    tr = {"text": "hi", "usage": {"type": "duration", "seconds": 90}}
    assert estimate_from_response(tr, model="whisper-1").usage == {"audio_seconds": 90.0}


def test_sources_are_listed_and_open():
    src = {r["source_id"]: r for r in rows("modality_sources.csv")}
    for r in FACTORS.values():
        for s in r["source_ids"].split(";"):
            assert src[s]["role"] != "context only", (r["functional_unit"], s)
            assert src[s]["share_alike"] == "N"
    text = (ROOT / "modality_sources.csv").read_text(encoding="utf-8") + (ROOT / "modality_factors.csv").read_text(encoding="utf-8")
    assert "Users" not in text
    assert {r["basis"] for r in FACTORS.values()} <= {"derived", "assumption"}


def test_cli_modality():
    p = subprocess.run([sys.executable, "-m", "saef", "modality", "--model", "dall-e-3", "--unit", "images", "--count", "10", "--region", "eu-west-1"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "image_generation_diffusion" in p.stdout and "water mL" in p.stdout
    p = subprocess.run([sys.executable, "-m", "saef", "estimate", "--model", "text-embedding-3-small", "--tokens", "1000"], capture_output=True, text=True)
    assert p.returncode == 0 and "embedding_text" in p.stdout
    p = subprocess.run([sys.executable, "-m", "saef", "modality", "--model", "lyria-2", "--unit", "images", "--count", "1"], capture_output=True, text=True)
    assert p.returncode == 2
