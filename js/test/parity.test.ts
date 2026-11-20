import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SaefError, estimate, estimateFromResponse, factor, grids, modelClassFor, regions, citation, DATASET_VERSION, VERSION } from "../src/index.js";

const P = JSON.parse(readFileSync(new URL("./parity.json", import.meta.url), "utf-8"));

const close = (got: number | null, want: number | null, label: string) => {
  if (want === null) return expect(got, label).toBeNull();
  expect(got, label).not.toBeNull();
  const g = got as number;
  const tol = 1e-12 * Math.max(Math.abs(want), Math.abs(g));
  expect(Math.abs(g - want) <= tol, `${label}: ${g} vs ${want}`).toBe(true);
};

describe("parity with the Python package", () => {
  it("versions match", () => {
    expect(VERSION).toBe(P.version);
    expect(DATASET_VERSION).toBe(P.dataset_version);
    expect(citation()).toBe(P.citation);
  });

  it("estimate() on a grid of models, regions and token splits", () => {
    let ok = 0;
    let errors = 0;
    for (const c of P.cases) {
      const label = JSON.stringify({ ...c, expected: undefined });
      const args =
        c.kind === "split"
          ? { model: c.model, region: c.region, inputTokens: c.input_tokens, outputTokens: c.output_tokens, cachedInputTokens: c.cached_input_tokens }
          : { model: c.model, region: c.region, tokens: c.tokens };
      if (c.expected === null) {
        expect(() => estimate(args), label).toThrow(SaefError);
        errors++;
        continue;
      }
      const e = estimate(args);
      const x = c.expected;
      close(e.kgCo2e, x.kg_co2e, `${label} kgCo2e`);
      close(e.kgCo2eHighCase, x.kg_co2e_high_case, `${label} high`);
      close(e.waterMl, x.water_ml, `${label} waterMl`);
      close(e.litres, x.water_ml === null ? null : x.water_ml / 1000, `${label} litres`);
      expect(e.modelClass, label).toBe(x.model_class);
      expect(e.region, label).toBe(x.region_id);
      close(e.effectiveTokens, x.effective, `${label} effective`);
      close(e.billedTokens, x.billed, `${label} billed`);
      if (c.kind === "split") {
        close(e.developmentLineKgCo2e, x.development_line_kg_co2e, `${label} dev`);
        close(e.kwh, x.kwh, `${label} kwh`);
        close(e.waterMlExclHydro, x.water_ml_excl_hydro, `${label} waterExcl`);
        for (const k of ["electricity", "embodied", "construction", "training"] as const) {
          close(e.termsKgCo2e[k], x.terms[k], `${label} term ${k}`);
        }
        expect(e.classBasis, label).toBe(x.class_basis);
        expect(e.warnings, label).toEqual(x.warnings);
      }
      expect(e.factorVersion).toBe("3.1.0");
      ok++;
    }
    expect(ok).toBeGreaterThan(1000);
    expect(errors).toBeGreaterThan(50);
  });

  it("estimateFromResponse() on OpenAI Chat, OpenAI Responses and Anthropic shapes", () => {
    for (const c of P.responses) {
      const e = estimateFromResponse(c.response, { region: c.region });
      const x = c.expected;
      const label = `${c.name} ${c.region}`;
      close(e.kgCo2e, x.kg_co2e, label);
      close(e.kgCo2eHighCase, x.kg_co2e_high_case, label);
      close(e.waterMl, x.water_ml, label);
      close(e.billedTokens, x.billed, label);
      close(e.effectiveTokens, x.effective, label);
      expect(e.modelClass).toBe(x.model_class);
      expect(e.region).toBe(x.region_id);
    }
  });

  it("class factors, model classes, region and grid lists", () => {
    for (const f of P.factors) {
      close(factor(f.model_class, f.region === "global-average" ? null : f.region).kgCo2ePer1mBilledTokens, f.factor, `${f.model_class} ${f.region}`);
    }
    for (const [m, cl] of Object.entries(P.model_class)) {
      if (cl === null) expect(() => modelClassFor(m)).toThrow(SaefError);
      else expect(modelClassFor(m), m).toBe(cl);
    }
    expect(regions()).toEqual(P.regions);
    expect(grids()).toEqual(P.grids);
  });
});
