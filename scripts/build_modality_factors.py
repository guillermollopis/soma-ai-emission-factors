"""Build modality_factors.csv: energy, CO2e and water per functional unit for the non-text model types.

Every value is stored at GPU (accelerator) level and then goes through the text chain unchanged:
    server = GPU x H x I, facility = server x PUE, CO2e = facility x grid + lifecycle per server kWh, water as for text.
Inputs are written out with their source ids (modality_sources.csv). Standard library only.
Run: python -I scripts/build_modality_factors.py
"""
import csv
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(name):
    with open(ROOT / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


LIFE = {(r["model_class"], r["region_id"]): r for r in read("factors_v3_lifecycle.csv") if r["gpu"] == "H100"}
TOK = {r["constant"]: r for r in read("token_basis_constants.csv")}
MODEL = read("model_factors.csv")[0]

H, I, PUE = (float(LIFE["A", "eu-west-1"][k]) for k in ("h_host_uplift", "i_idle_uplift", "pue"))
H_HI, I_HI, PUE_HI = (float(MODEL[k]) for k in ("h_host_uplift_high_case", "i_idle_uplift_high_case", "pue_high_case"))
W_IN, W_CACHE, D = (float(TOK[k]["central"]) for k in ("W_in", "W_cache", "D"))

# Six-significant-figure class values of the v3.3.0 deposit file term_constants.csv (H100).
CLASS_GPU = {"A": 0.0363501, "B": 0.119037, "C": 0.443268}
A_SERVER, A_SERVER_HI = 0.0599777, 0.0820058
A_EMB, A_CON, A_EMB_HI, A_CON_HI = 0.000887014, 0.000459429, 0.00110877, 0.000628165
# kg per million effective tokens / kWh per million effective tokens (Wh per 1k is kWh per million)
EMB_PER_KWH, CON_PER_KWH = A_EMB / A_SERVER, A_CON / A_SERVER
EMB_HI_PER_KWH, CON_HI_PER_KWH = A_EMB_HI / A_SERVER_HI, A_CON_HI / A_SERVER_HI


def member(cls, which):
    r = LIFE[cls, "eu-west-1"]
    return float(r[f"gpu_j_per_output_token_{which}_member"]) / float(r["gpu_j_per_output_token"])


CLS = {c: (v, v * member(c, "lowest"), v * member(c, "highest")) for c, v in CLASS_GPU.items()}
for c, v in CLASS_GPU.items():
    assert abs(v - float(LIFE[c, "eu-west-1"]["gpu_wh_per_1k_effective_tokens"])) < 6e-6

rows = []


def add(model_type, cls, unit, central, low, high, basis, sources, notes):
    assert low <= central <= high, (model_type, cls, unit, low, central, high)
    rows.append((model_type, cls, unit, central, low, high, basis, sources, notes))


def tok(cls, effective_per_token):
    """GPU Wh per 1,000 tokens of one kind: class value, lowest and highest measured member."""
    return tuple(x * effective_per_token for x in CLS[cls])


OUT, INP, CACHED = 1 / D, W_IN / D, W_CACHE / D

# ------------------------------------------------------------------ release 1 (SAEF v3.4.0)
A_IN = tok("A", INP)
B_IN = tok("B", INP)
# MLPerf: whole-server wall power, so the GPU-equivalent is wall / H (the chain multiplies H back in).
BERT = 5080 / 53726.9 / 384 * 1000 / 3600 / H                       # Wh per 1k tokens
SPEECH_S = (5.4 + 5.3) * 3600 / 1633
WHISPER_LOW = 7636.49 / 688.623 / SPEECH_S * 60 / 3600 / H          # Wh per audio minute
# AI Energy Score: GPU energy, H100, batch 1, FP32. Wh per query / tokens per query.
E5 = 0.000602341 / 22.919 * 1000
SFR = 0.005220944 / 22.919 * 1000
BERT_TC = 0.000918014 / 156.925 * 1000
VIT = 0.001163123
WHISPER_HIGH = 0.086679066 * 2                                      # two padded 30-s windows per minute
A_ACTIVE_B = (8 + 8 + 9 + 12 + 12 + 14 + 3) / 7
EMB_S, RR_S = A_IN[0] * 0.335 / A_ACTIVE_B, A_IN[0] * 0.56 / A_ACTIVE_B
L_LOW, RR_LOW = BERT * 7 / 0.335, BERT * 0.56 / 0.335
ENC = "R1-S01;R1-S02;R1-S05;R1-S06"

add("embedding_text", "E-small", "1k_input_tokens", EMB_S, BERT, E5, "derived", ENC,
    "central: class A input-token energy scaled by active parameters 0.335B/9.43B (assumption: linear). low: MLPerf BERT-large H100 INT8, wall power / H, 384 tokens per sample. high: AI Energy Score e5-large-v2, batch 1 FP32 (unbatched upper bound)")
add("embedding_text", "E-large", "1k_input_tokens", A_IN[0], L_LOW, SFR, "derived", ENC,
    "central: LLM-based embedder treated as class A prefill (W_in/D x class A). low: BERT-large low scaled by 7B/0.335B (assumption). high: AI Energy Score SFR-Embedding-Mistral 7B, batch 1 FP32")
add("embedding_image", "E-small", "image", EMB_S * 0.257, BERT * 0.257, VIT, "derived", ENC,
    "image = 257 patch tokens (ViT-L/14 at 224 px, assumption) x E-small. high: AI Energy Score vit-large-patch32-384, batch 1 FP32")
add("embedding_image", "E-large", "image", A_IN[0], L_LOW * 0.256, SFR, "assumption", ENC,
    "image = 1,000 token equivalents (assumption, 256 in the low row) x E-large")
for cls, (c, lo, hi), note in (
    ("R-small", (RR_S, RR_LOW, BERT_TC), "cross-encoder, class A input-token energy scaled to 0.56B (assumption: linear). high: AI Energy Score bert-large-sst2, batch 1 FP32"),
    ("R-large", (A_IN[0], L_LOW, SFR), "LLM-based reranker: same values as E-large"),
):
    add("rerank", cls, "1k_input_tokens", c, lo, hi, "derived", ENC,
        note + ". Tokens scored = query + document, summed over all pairs")
    add("rerank", cls, "search_unit", c * 30, lo * 5, hi * 50, "assumption", ENC + ";R1-S14",
        "use only when tokens are unknown: 30,000 tokens per search unit (5,000 to 50,000, the billing cap of 100 documents x 500 tokens)")
for cls, (c, lo, hi), src, note in (
    ("M-A", A_IN, "R1-S01", "small-tier moderation model: class A prefill, low and high measured member"),
    ("M-B", (B_IN[0], A_IN[0], B_IN[2]), "R1-S01;R1-S15", "omni-moderation, described by OpenAI as built on GPT-4o: class B prefill. low: class A. high: highest class B member"),
):
    add("moderation", cls, "1k_input_tokens", c, lo, hi, "derived", src, note)
    add("moderation", cls, "call", c * 0.5, lo * 0.021, hi * 2.5, "assumption", src + ";R1-S16",
        "use only when text length is unknown: 500 tokens per call (21 to 2,500, WildChat-1M statistics)")
    add("moderation", cls, "1k_text_output_tokens", *tok(cls[-1], OUT), "derived", "R1-S01",
        "only for guard models that bill generated tokens: text output of the class")

J = 1 / 3600
D_SMALL = [417.1, 714.2, 1181.1, 1581.3]       # GPU J per 1024x1024 image: SANA 1.5 1.6B, PixArt-Sigma, SANA 1.5 4.8B, SD3.5 medium
D_LARGE = [4420.1, 5180.9, 7948.6]             # SD3.5 large, HunyuanDiT, FLUX.1-dev
DS, DL = ((sum(v) / len(v) * J, min(v) * J, max(v) * J) for v in (D_SMALL, D_LARGE))
add("image_generation_diffusion", "D-small", "image", *DS, "derived", "R1-S04",
    "mean of four open models; low and high are the lowest and highest GPU energy per 1024x1024 image measured by ML.ENERGY (SANA 1.5 1.6B, PixArt-Sigma, SANA 1.5 4.8B, SD3.5 medium; 20 to 28 steps). Linear in megapixels above 1.05 MP (assumption)")
add("image_generation_diffusion", "D-large", "image", *DL, "derived", "R1-S04",
    "mean of three open models; low and high are the lowest and highest GPU energy per 1024x1024 image measured by ML.ENERGY (SD3.5 large, HunyuanDiT, FLUX.1-dev). Default for closed per-image models. Linear in megapixels above 1.05 MP (assumption)")

IMG = "R1-S01;R1-S04"
for cls in "ABC":
    name = f"IMG-TOK-{cls}"
    p = tok(cls, OUT)
    p = (p[0], p[1], max(p[2] if cls == "C" else 0, tok("C", OUT)[0]))
    add("image_generation_token_billed", name, "1k_image_output_tokens_parity", *p, "assumption", "R1-S01",
        f"one image output token charged as one class {cls} output token (1/D effective tokens; assumption). high: class C. Only for use with an image count, as max(tokens x this row, images x D-small)")
    add("image_generation_token_billed", name, "1k_image_output_tokens", max(p[0], DS[0] / 1.056), max(p[1], DS[1] / 4.160),
        max(p[2], DS[0] / 0.272), "assumption", IMG + ";R1-S10",
        "image count unknown: max(token parity, D-small per image at 1,056 tokens per image). low at 4,160 and high at 272 tokens per image. The calculator charges at least one D-small image per record. Pass the image count for a tighter value")
    for unit, n, note in (("image_1024_low", 272, "272 tokens"), ("image_1024_medium", 1056, "1,056 tokens"), ("image_1024_high", 4160, "4,160 tokens")):
        add("image_generation_token_billed", name, unit, *(max(p[i] * n / 1000, DS[i]) for i in range(3)), "assumption", IMG + ";R1-S10",
            f"max(token parity, D-small per 1024x1024 image). {note} (OpenAI table for GPT Image 1, 1.5 and 1 mini; an assumption for gpt-image-2 and later, which report usage tokens)")
    add("image_generation_token_billed", name, "image", max(p[0] * 1.12, DS[0]), max(p[1] * 0.272, DS[1]), max(p[2] * 4.16, DS[2]), "assumption", IMG + ";R1-S10;R1-S11",
        "image count only, size and quality unknown: max(token parity at 1,120 tokens, D-small per image). low at 272 and high at 4,160 tokens")
    add("image_generation_token_billed", name, "image_4k", max(p[0] * 2, DS[0]), max(p[1] * 2, DS[1]), max(p[2] * 2, DS[0] * 16), "assumption", IMG + ";R1-S11",
        "4K image at 2,000 tokens: max(token parity, D-small per 1024x1024 image), assuming the 4K output is upscaled. high: D-small scaled linearly to 16 times the pixels")
    add("image_generation_token_billed", name, "1k_input_tokens", *tok(cls, INP), "derived", "R1-S01", "text and image input tokens as uncached text input of the class")
    add("image_generation_token_billed", name, "1k_cached_input_tokens", *tok(cls, CACHED), "derived", "R1-S01", "cached input tokens as cached text input of the class")
    add("image_generation_token_billed", name, "1k_text_output_tokens", *tok(cls, OUT), "derived", "R1-S01", "text output tokens as text output of the class")

add("speech_to_text", "STT-ENC", "audio_minute", math.sqrt(WHISPER_LOW * WHISPER_HIGH), WHISPER_LOW, WHISPER_HIGH, "derived", "R1-S03;R1-S05;R1-S06;R1-S09;R1-S17",
    "Whisper family. low: MLPerf v5.1 whisper-large-v3 on 8 B200, wall power / H. high: AI Energy Score whisper-large-v3, batch 1 FP32, two padded 30-s windows per minute. central: geometric mean (assumption). Clips much shorter than 30 s can exceed the high row")
for cls in "AB":
    name = f"STT-LLM-{cls}"
    eff = lambda audio: (200 + W_IN * audio) / D / 1000
    add("speech_to_text", name, "audio_minute", CLS[cls][0] * eff(1000), CLS[cls][1] * eff(600), CLS[cls][2] * eff(1600), "assumption", "R1-S01;R1-S12",
        "use only when tokens are unknown: 1,000 audio input tokens per minute (600 to 1,600) + 200 transcript tokens (assumption)")
    add("speech_to_text", name, "1k_input_tokens", *tok(cls, INP), "derived", "R1-S01", "audio and text input tokens as uncached text input of the class")
    add("speech_to_text", name, "1k_cached_input_tokens", *tok(cls, CACHED), "derived", "R1-S01", "cached input tokens as cached text input of the class")
    add("speech_to_text", name, "1k_text_output_tokens", *tok(cls, OUT), "derived", "R1-S01", "transcript tokens as text output of the class")

# ------------------------------------------------------------------ chain and reference regions
WATER = {r["region_id"]: r for r in read("region_water_factors.csv")}
REGIONS = ("eu-west-1", "us-east-1", "global-average")
GRID = {r["region_id"]: float(r["grid_kg_co2e_per_kwh"]) for r in read("cloud_region_factors.csv")}
GRID["global-average"] = float(next(r["grid_kg_co2e_per_kwh"] for r in read("grid_intensity_all.csv") if r["grid_id"] == "WORLD"))
CASES = ("central", "low", "high")


def s(x):
    return f"{x:.6g}"


def water_ml(server_wh, facility_wh, rid, ewif_col):
    w = WATER[rid]
    onsite = (server_wh if w["wue_denominator"].startswith("IT load") else facility_wh) * float(w["wue_l_per_kwh"])
    if w["wue_basis_as_published"] == "withdrawal":
        onsite *= float(w["withdrawal_to_consumption"])
    return onsite + facility_wh * float(w[ewif_col])


out = []
for model_type, cls, unit, c, lo, hi, basis, sources, notes in rows:
    r = {"model_type": model_type, "class": cls, "functional_unit": unit}
    gpu = dict(zip(CASES, (c, lo, hi)))
    server = {k: v * H * I for k, v in gpu.items()}
    fac = {k: v * PUE for k, v in server.items()}
    server_hc = c * H_HI * I_HI
    for label, d in (("gpu_wh", gpu), ("server_wh", server), ("facility_wh", fac)):
        r.update({f"{label}_{k}": s(d[k]) for k in CASES})
    r["facility_wh_high_case"] = s(server_hc * PUE_HI)
    r.update({f"embodied_g_co2e_{k}": s(server[k] * EMB_PER_KWH) for k in CASES})
    r.update({f"construction_g_co2e_{k}": s(server[k] * CON_PER_KWH) for k in CASES})
    r["embodied_g_co2e_high_case"] = s(server_hc * EMB_HI_PER_KWH)
    r["construction_g_co2e_high_case"] = s(server_hc * CON_HI_PER_KWH)
    for rid in REGIONS:
        col = rid.replace("-", "_")
        for k in CASES:
            r[f"co2e_g_{col}_{k}"] = s(fac[k] * GRID[rid] + server[k] * (EMB_PER_KWH + CON_PER_KWH))
        r[f"co2e_g_{col}_high_case"] = s(server_hc * (PUE_HI * GRID[rid] + EMB_HI_PER_KWH + CON_HI_PER_KWH))
    for rid in REGIONS[:2]:
        col = rid.replace("-", "_")
        for label, ew in (("water_ml", "ewif_l_per_kwh"), ("water_ml_excl_hydro", "ewif_excl_hydro_l_per_kwh")):
            r.update({f"{label}_{col}_{k}": s(water_ml(server[k], fac[k], rid, ew)) for k in CASES})
    r.update(basis=basis, source_ids=sources, notes=notes)
    out.append(r)

assert len({(r["model_type"], r["class"], r["functional_unit"]) for r in out}) == len(out)
with open(ROOT / "modality_factors.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(out)
print(f"modality_factors.csv: {len(out)} rows, chain H {H} I {I} PUE {PUE}")
