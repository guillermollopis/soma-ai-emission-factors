import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SaefError, estimateModality, estimateModalityUsage } from "../src/index.js";

const CASES = JSON.parse(readFileSync(new URL("./modality-parity.json", import.meta.url), "utf-8"));

const camel = (s: string): string => s.replace(/_([a-z0-9])/g, (_m, c: string) => c.toUpperCase());
const RENAME: Record<string, string> = { regionId: "region", version: "factorVersion", class: "modelClass" };
const SKIP = new Set(["provenance", "water", "effective_tokens"]);

const same = (got: unknown, want: unknown, label: string) => {
  if (typeof want === "number") {
    expect(typeof got, label).toBe("number");
    const g = got as number;
    const tol = 1e-9 * Math.max(Math.abs(want), Math.abs(g));
    expect(Math.abs(g - want) <= tol, `${label}: ${g} vs ${want}`).toBe(true);
  } else if (Array.isArray(want)) {
    expect(Array.isArray(got), label).toBe(true);
    expect((got as unknown[]).length, label).toBe(want.length);
    want.forEach((w, i) => same((got as unknown[])[i], w, `${label}[${i}]`));
  } else if (want !== null && typeof want === "object") {
    const g = got as Record<string, unknown>;
    for (const [k, w] of Object.entries(want as Record<string, unknown>)) {
      if (SKIP.has(k)) continue;
      const key = RENAME[camel(k)] ?? camel(k);
      if (key === "usage") {
        const u = g.usage as Record<string, number>;
        expect(Object.keys(u).sort(), label).toEqual(Object.keys(w as object).sort());
        for (const [unit, n] of Object.entries(w as Record<string, number>)) same(u[unit], n, `${label}.usage.${unit}`);
        continue;
      }
      same(g[key], w, `${label}.${key}`);
    }
  } else {
    expect(got, label).toEqual(want);
  }
};

const run = (c: Record<string, any>) => {
  const opts = {
    region: c.region,
    provider: c.provider,
    megapixels: c.megapixels,
    modelType: c.model_type,
    modelClass: c.model_class,
  };
  if (c.usage) return estimateModalityUsage(c.usage, c.model, opts);
  return estimateModality(c.unit_count === "Infinity" ? Infinity : c.unit_count, c.unit, c.model, opts);
};

describe("estimateModality parity with the Python package", () => {
  it("covers every model type, class and unit", () => {
    const ok = CASES.filter((c: any) => c.expected.result);
    expect(ok.length).toBeGreaterThanOrEqual(50);
    const types = new Set(ok.map((c: any) => c.expected.result.model_type));
    const TYPES = [
      "embedding_text", "embedding_image", "rerank", "moderation", "image_generation_diffusion",
      "image_generation_token_billed", "speech_to_text", "tts", "realtime_turn_based", "realtime_full_duplex",
      "video_generation",
    ];
    for (const t of TYPES) expect(types.has(t), t).toBe(true);
    const units = new Set(ok.flatMap((c: any) => c.expected.result.components.map((x: any) => x.unit)));
    for (const u of [
      "characters", "audio_output_tokens", "user_audio_minutes", "assistant_audio_minutes", "context_minutes",
      "conversation_minutes", "active_session_minutes", "video_seconds", "video_seconds_480p", "video_seconds_720p",
      "video_seconds_1080p", "video_seconds_2160p", "video_clips",
    ]) expect(units.has(u), u).toBe(true);
    expect(ok.some((c: any) => c.expected.result.warnings.some((w: string) => w.startsWith("video values are indicative")))).toBe(true);
    expect(CASES.some((c: any) => c.expected.error)).toBe(true);
  });

  CASES.forEach((c: any, i: number) => {
    const label = `#${i} ${JSON.stringify({ ...c, expected: undefined })}`;
    it(label, () => {
      if (c.expected.error !== undefined) {
        let err: unknown;
        try {
          run(c);
        } catch (e) {
          err = e;
        }
        expect(err, label).toBeInstanceOf(SaefError);
        const msg = (err as Error).message;
        if (c.expected.error.startsWith("Unknown region")) {
          expect(msg, label).toContain(c.expected.error.split(".")[0] + ".");
        } else {
          expect(msg, label).toBe(c.expected.error);
        }
        return;
      }
      const got = run(c);
      const want = c.expected.result;
      same(got, want, label);
      expect(got.warnings, label).toEqual(want.warnings);
      expect(got.note).toBe(want.note);
    });
  });
});
