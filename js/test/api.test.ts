import { describe, expect, it } from "vitest";
import { mapModel } from "../src/calc.js";
import { SaefError, Tracker, estimate, estimateFromResponse, usageFromResponse } from "../src/index.js";

describe("api", () => {
  it("rejects bad input", () => {
    expect(() => estimate({ model: "gpt-4o" })).toThrow(SaefError);
    expect(() => estimate({ model: "gpt-4o", tokens: 1, outputTokens: 1 })).toThrow(SaefError);
    expect(() => estimate({ model: "gpt-4o", outputTokens: -1 })).toThrow(SaefError);
    expect(() => estimate({ model: "gpt-4o", outputTokens: 1, region: "nope-region-9" })).toThrow(/Unknown region/);
  });

  it("returns water in litres only where the region has a factor", () => {
    const eu = estimate({ model: "gpt-4o", inputTokens: 1000, outputTokens: 500, region: "eu-north-1" });
    expect(eu.litres).toBeGreaterThan(0);
    expect(eu.litres).toBeCloseTo((eu.waterMl as number) / 1000, 15);
    expect(eu.kwh).toBeGreaterThan(0);
    const world = estimate({ model: "gpt-4o", inputTokens: 1000, outputTokens: 500 });
    expect(world.litres).toBeNull();
    expect(world.region).toBe("global-average");
  });

  it("reads usage shapes", () => {
    expect(usageFromResponse({ model: "m", usage: { prompt_tokens: 10, completion_tokens: 5, prompt_tokens_details: { cached_tokens: 4 } } })).toEqual({
      model: "m", inputTokens: 6, cachedInputTokens: 4, outputTokens: 5,
    });
    expect(usageFromResponse({ model: "m", usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 3, cache_creation_input_tokens: 2 } })).toEqual({
      model: "m", inputTokens: 12, cachedInputTokens: 3, outputTokens: 5,
    });
    expect(() => usageFromResponse({ model: "m" })).toThrow(SaefError);
    expect(() => estimateFromResponse({ usage: { prompt_tokens: 1, completion_tokens: 1 } })).toThrow(/no model/);
  });

  it("Tracker adds, skips out-of-scope models, falls back on unknown per-call region, resets", () => {
    const t = new Tracker({ region: "eu-west-1" });
    const a = t.add({ model: "gpt-4o", inputTokens: 100, outputTokens: 50 });
    t.add({ model: "gpt-4o", inputTokens: 100, outputTokens: 50, cachedInputTokens: 10, region: "mars-1" });
    expect(t.add({ model: "text-embedding-3-small", inputTokens: 5, outputTokens: 0 })).toBeNull();
    const tot = t.totals();
    expect(tot.calls).toBe(2);
    expect(tot.skippedCalls).toBe(1);
    expect(tot.inputTokens).toBe(200);
    expect(tot.cachedInputTokens).toBe(10);
    expect(tot.byModel["gpt-4o"].calls).toBe(2);
    expect(tot.kgCo2e).toBeGreaterThan((a as { kgCo2e: number }).kgCo2e);
    t.reset();
    expect(t.totals().calls).toBe(0);
  });
});

describe("model mapping", () => {
  it("non-text and out-of-scope rows win over longer text prefixes, camelCase ids are split", () => {
    expect(mapModel("gpt-4o-mini-transcribe-2025-12-15").modelType).toBe("speech_to_text");
    expect(mapModel("gpt-4o-mini-tts").modelType).toBe("tts");
    expect(mapModel("gemini-omni-flash").modelClass).toBe("B");
    expect(mapModel("TitanEmbeddingV2-Text").modelType).toBe("embedding_text");
    expect(mapModel("TitanEmbeddingsG1-Image").modelClass).toBe("E-large");
    expect(mapModel("advanced-voice-4o").modelType).toBe("realtime_turn_based");
    expect(mapModel("gpt-4o-audio-preview").outOfScope).toBe(true);
    expect(mapModel("DeepSeekV3").modelClass).toBe("C");
    expect(mapModel("img-1p-p-ev3").isDefaultFallback).toBe(true);
    expect(() => estimate({ model: "gpt-image-2", outputTokens: 10 })).toThrow(/estimateModality/);
    expect(() => estimate({ model: "gpt-4o-audio-preview", outputTokens: 10 })).toThrow(/out of scope/);
  });
});
