"""Dump expected estimate_modality values from the Python package for the TypeScript parity test.

Usage (from the repo root): PYTHONPATH=$PWD/src python js/scripts/dump-modality-parity.py
"""
import json
from pathlib import Path

import saef
from saef import _calc, _modality

out_path = Path(__file__).resolve().parent.parent / "test" / "modality-parity.json"

example = {}
for r in _calc.read("model_class_mapping.csv"):
    if r["model_type"] == "text" or r["model_class"] == "out-of-scope" or r["basis"] == "default":
        continue
    for alt in r["pattern"].split(";"):
        alt = alt.strip()
        if alt and "*" not in alt:
            mm = _calc.map_model(alt)
            if mm["model_type"] == r["model_type"] and mm["model_class"] == r["model_class"]:
                example.setdefault((r["model_type"], r["model_class"]), alt)
                break

cases = []


def add(**kw):
    cases.append(kw)


regions = ["us-east-1", "eu-west-1", "global", "ESP"]
for i, ((t, c), model) in enumerate(sorted(example.items())):
    units = sorted({u for (tt, cc, u) in _modality._rows() if tt == t and cc == c})
    for u in sorted(n for n, (f, _s) in _modality.UNITS.items() if f in units):
        if u == "image_output_tokens":
            add(unit_count=3300, unit=u, model=model, region=regions[i % 4])
        else:
            add(unit_count=250 if "tokens" in u else 7, unit=u, model=model, region=regions[(i + len(u)) % 4])

for i, (t, c) in enumerate(sorted({(t, c) for t, c, _u in _modality._rows()})):
    fus = {u for (tt, cc, u) in _modality._rows() if tt == t and cc == c}
    units = sorted(n for n, (f, _s) in _modality.UNITS.items() if f in fus)
    for u in units if (t, c) not in example else units[:1]:
        add(unit_count=1200 if "tokens" in u else 3, unit=u, model_type=t, model_class=c.lower(), region=regions[i % 4])

add(unit_count=2000, unit="embedding_tokens", model="text-embedding-3-small", region="eu-west-1")
add(unit_count=2, unit="1k_input_tokens", model="text-embedding-3-small", region="us-east-1")
add(unit_count=90, unit="Audio-Seconds", model="whisper-1", region="global")
add(unit_count=5, unit="image", model="dall-e-3", region="us-east-1")
add(unit_count=5, unit="images", model="dall-e-3", region="us-east-1", megapixels=4.0)
add(unit_count=5, unit="images", model="dall-e-3", region="us-east-1", megapixels=0.5)
add(unit_count=500, unit="input_tokens", model="text-embedding-3-small", region="us-east-1", megapixels=2.0)
add(unit_count=3, unit="images", model="dall-e-3", region="us-east-1", provider="aws")
add(unit_count=3300, unit="image_output_tokens", model="gpt-image-1.5", region="us-east-1")
add(unit_count=0, unit="image_output_tokens", model="gpt-image-1.5", region="us-east-1")
add(unit_count=60, unit="images_1024_high", model="gpt-image-1.5", region="eu-west-1")

for model in ("gpt-image-1.5", "dall-e-3"):
    add(usage={"image_output_tokens": 3300, "images": 3}, model=model, region="us-east-1")
    add(usage={"image_output_tokens": 3300, "images": 3}, model=model, region="global")
add(usage={"input_tokens": 1000, "text_output_tokens": 400, "images": 2}, model="gpt-image-1.5", region="eu-west-1")
add(usage={"input_tokens": 1000, "audio_minutes": 2}, model="gpt-4o-transcribe", region="us-east-1")
add(usage={"input_tokens": 1000, "input_tokens ": 500}, model="text-embedding-3-small", region="ESP")

add(unit_count=10, unit="images", model="gpt-4o", region="us-east-1")
add(unit_count=10, unit="calls", model="claude-sonnet-4-5", region="global")
add(unit_count=10, unit="search_units", model="totally-unknown-model-xyz", region="eu-west-1")
add(unit_count=10, unit="audio_minutes", model="text-embedding-3-small", region="us-east-1")
add(unit_count=10, unit="images", model="whisper-1", region="us-east-1")
add(unit_count=10, unit="audio_minutes", model="rerank-v3.5", region="us-east-1")
add(unit_count=10, unit="images_4k", model="gpt-4o", region="global")
add(unit_count=100, unit="input_tokens", model_type="embedding_text", model_class="e-large", region="eu-west-1")
add(unit_count=100, unit="input_tokens", model_type="embedding_text", model_class="E-small", region="global")
add(unit_count=2, unit="images", model_type="image_generation_diffusion", model_class="d-large", region="us-east-1", megapixels=2.1)

add(unit_count=4500, unit="characters", model="tts-1", region="eu-west-1")
add(unit_count=3, unit="1k_characters", model="eleven_multilingual_v2", region="global")
add(unit_count=2, unit="audio_minutes", model="gpt-4o-mini-tts", region="us-east-1")
add(usage={"input_tokens": 300, "audio_output_tokens": 2400}, model="gpt-4o-mini-tts", region="us-east-1")
add(unit_count=1.5, unit="1k_audio_output_tokens", model="gpt-realtime", region="ESP")
add(usage={"input_tokens": 800, "cached_input_tokens": 4000, "text_output_tokens": 50, "audio_output_tokens": 900},
    model="gpt-realtime-mini", region="eu-west-1")
add(usage={"user_audio_minutes": 4, "assistant_audio_minutes": 3, "context_minutes": 6}, model="gpt-realtime", region="us-east-1")
add(unit_count=12, unit="conversation_minutes", model="advanced-voice", region="global")
add(unit_count=12, unit="conversation_minute", model="nova-sonic", region="us-east-1")
add(usage={"conversation_minutes": 5, "audio_output_tokens": 100}, model="gpt-realtime", region="us-east-1")
add(usage={"conversation_minutes": 5, "context_minutes": 1}, model="gpt-realtime", region="us-east-1")
add(unit_count=30, unit="active_session_minutes", model="gpt-live", region="eu-west-1")
add(unit_count=30, unit="active-session-minute", model="advanced-voice-bidi", region="global")
add(unit_count=30, unit="input_tokens", model="gpt-live", region="global")
for j, u in enumerate(("video_seconds", "video_seconds_480p", "video_seconds_720p", "video_seconds_1080p", "video_seconds_2160p", "video_clips")):
    for k, model in enumerate(("sora-2", "sora-2-pro", "veo-3.1", "veo-3.1-fast", "veo-2", "kling-2.1-pro", "wan2.1-1.3b")):
        add(unit_count=8 if u != "video_clips" else 2, unit=u, model=model, region=regions[(j + k) % 4])
add(unit_count=8, unit="Video-Second-1080p", model="veo-3.1", region="us-east-1")
add(unit_count=8, unit="video_second", model="veo-3.1", region="us-east-1")
add(unit_count=1, unit="video_clip", model="sora-2", region="us-east-1")
add(unit_count=8, unit="video_seconds", model="veo-3.1", region="us-east-1", megapixels=2.0)
add(usage={"video_seconds_1080p": 8, "video_clips": 1}, model="veo-3.1", region="eu-west-1")
add(usage={"video_seconds": 8, "images": 1}, model="veo-3.1", region="us-east-1")
add(unit_count=8, unit="video_seconds", model="gemini-omni-flash", region="us-east-1")
add(unit_count=8, unit="video_seconds_1080p", model="google/gemini-omni-flash-preview", region="eu-west-1")
add(usage={"video_seconds": 8, "input_tokens": 1000}, model="gemini-omni-flash", region="us-east-1")
add(unit_count=1000, unit="input_tokens", model="gemini-omni-flash", region="us-east-1")
add(unit_count=1000, unit="audio_output_tokens", model="gemini-omni-flash", region="us-east-1")
add(unit_count=4, unit="video_clips", model="gemini-omni-video", region="global")
add(unit_count=6, unit="video_seconds_2160p", model="gpt-4o", region="us-east-1")
add(unit_count=6, unit="video_seconds", model="totally-unknown-model-xyz", region="global")
add(unit_count=600, unit="characters", model="claude-sonnet-4-5", region="us-east-1")
add(unit_count=600, unit="characters", model="whisper-1", region="us-east-1")
add(unit_count=500, unit="audio_output_tokens", model="totally-unknown-model-xyz", region="eu-west-1")
add(unit_count=5, unit="active_session_minutes", model="gpt-4o", region="eu-west-1")
add(unit_count=5, unit="user_audio_minutes", model="gpt-4o", region="eu-west-1")
add(unit_count=5, unit="context_minutes", model="text-embedding-3-small", region="us-east-1")
add(unit_count=5, unit="video_seconds", model="tts-1", region="us-east-1")
add(unit_count=5, unit="characters", model="sora-2", region="us-east-1")
add(unit_count=5, unit="images", model="veo-3.1", region="us-east-1")
add(unit_count=5, unit="video_seconds", model_type="tts", model_class="TTS-B", region="us-east-1")
add(unit_count=5, unit="characters", model_type="video_generation", model_class="V-B", region="us-east-1")
add(unit_count=5, unit="video_seconds", model_type="video_generation", model_class="V-Z", region="us-east-1")
add(unit_count=5, unit="input_tokens", model="sora-2", region="us-east-1")
add(unit_count=5, unit="audio_minutes", model="gpt-4o-audio-preview", region="us-east-1")
add(unit_count=10, unit="images", model="gpt-4o-mini-tts", region="us-east-1")
add(unit_count=10, unit="tokens", model="text-embedding-3-small", region="us-east-1")
add(unit_count=-1, unit="input_tokens", model="text-embedding-3-small", region="us-east-1")
add(unit_count=float("inf"), unit="input_tokens", model="text-embedding-3-small", region="us-east-1")
add(unit_count=10, unit="input_tokens", model="text-embedding-3-small", region="nowhere-9")
add(unit_count=10, unit="input_tokens", model="text-embedding-3-small", region="us-east-1", megapixels=-1, model_type="x")
add(unit_count=10, unit="images", model="dall-e-3", region="us-east-1", megapixels=0)
add(unit_count=10, unit="input_tokens", model_type="embedding_text", model_class="Z", region="us-east-1")
add(unit_count=10, unit="input_tokens", model_type="embedding_text", region="us-east-1")
add(usage={}, model="text-embedding-3-small", region="us-east-1")
add(usage={"input_tokens": 5, "images_4k": 1, "image_output_tokens": 10}, model="gpt-image-1.5", region="us-east-1")
add(unit_count=10, unit="input_tokens", model="gpt-4o", region="us-east-1")


def run(c):
    c = dict(c)
    try:
        if "usage" in c:
            usage = c.pop("usage")
            r = _modality.estimate(c.pop("model", None), usage, c.pop("region", None), c.pop("provider", None),
                                   c.pop("megapixels", None), c.pop("model_type", None), c.pop("model_class", None),
                                   saef.DATASET_VERSION)
        else:
            r = saef.estimate_modality(c.get("model"), c["unit_count"], c["unit"], c.get("region"), c.get("provider"),
                                       c.get("megapixels"), c.get("model_type"), c.get("model_class"))
    except saef.SaefError as e:
        return {"error": str(e)}
    d = r.to_dict()
    for k in ("provenance", "water"):
        d.pop(k)
    return {"result": d}


for c in cases:
    if c.get("unit_count") == float("inf"):
        c["unit_count"] = "Infinity"
    c["expected"] = run({**c, "unit_count": float("inf")} if c.get("unit_count") == "Infinity" else c)

out_path.write_text(json.dumps(cases, indent=1, ensure_ascii=False), encoding="utf-8")
errors = sum("error" in c["expected"] for c in cases)
print(out_path, len(cases), "cases,", errors, "errors")
