import {
  DATASET_DOI,
  DATASET_VERSION,
  DATASET_VERSION_DOI,
  PAPER_DOI,
  SaefError,
  effectiveFromBilled,
  emissions,
  listGrids,
  listRegions,
  mapModel,
  water,
} from "./calc.js";

export { SaefError, DATASET_VERSION, DATASET_DOI, DATASET_VERSION_DOI, PAPER_DOI };

export const VERSION = "3.9.0";

const BIBTEX = `@misc{llopis2026somafactors,
  author       = {Llopis-Garcia, Guillermo},
  title        = {{SOMA AI Emission Factors (SAEF)}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3.5.0},
  doi          = {10.5281/zenodo.20443585},
  url          = {https://doi.org/10.5281/zenodo.20443585}
}

@article{llopis2026accounting,
  author  = {Llopis-Garcia, Guillermo},
  title   = {Accounting for purchased artificial intelligence inference in corporate greenhouse gas inventories: A token-based method},
  journal = {Sustainable Production and Consumption},
  volume  = {70},
  pages   = {320--347},
  year    = {2026},
  doi     = {10.1016/j.spc.2026.11.003},
  url     = {https://doi.org/10.1016/j.spc.2026.11.003}
}
`;

const NOTE =
  "Estimate, not a measurement. The high case is a conservative bound; the development line is reported separately and is not part of kgCo2e.";

export interface EstimateInput {
  model?: string;
  modelClass?: "A" | "B" | "C" | string;
  /** Billed total (input plus output). Use instead of outputTokens for a billed total without a split. */
  tokens?: number;
  /** Uncached input tokens. Cache reads go in cachedInputTokens, not here. */
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  /** Cloud region id, ISO-3 country code, eGRID subregion, 'eu-27', or omit for the global average. */
  region?: string | null;
  /** Cloud provider ('aws', 'azure', 'gcp') to disambiguate a region id shared by several providers. */
  provider?: string | null;
}

export interface Estimate {
  kgCo2e: number;
  kgCo2eHighCase: number;
  developmentLineKgCo2e: number;
  termsKgCo2e: { electricity: number; embodied: number; construction: number; training: number };
  /** Facility energy in kWh (electricity term divided by the grid intensity), null if the grid intensity is zero. */
  kwh: number | null;
  /** Water in litres under the hydropower-allocated (reig) convention, null where the region has no water factor. */
  litres: number | null;
  litresExclHydro: number | null;
  waterMl: number | null;
  waterMlExclHydro: number | null;
  model: string | null;
  modelClass: string;
  classBasis: string;
  region: string;
  gridKgCo2ePerKwh: number;
  billedTokens: number;
  effectiveTokens: number;
  warnings: string[];
  factorVersion: string;
  note: string;
}

const isNil = (v: unknown): v is null | undefined => v === null || v === undefined;

export function estimate(a: EstimateInput): Estimate {
  if (isNil(a.tokens) === isNil(a.outputTokens)) {
    throw new SaefError("Pass either tokens (billed total) or outputTokens (with inputTokens)");
  }
  const given = a.tokens ?? (a.outputTokens as number);
  const inp = a.inputTokens ?? 0;
  const cached = a.cachedInputTokens ?? 0;
  if (given < 0 || inp < 0 || cached < 0) throw new SaefError("token counts must be >= 0");
  const r = emissions({
    model: a.model,
    modelClass: a.modelClass,
    provider: a.provider,
    region: a.region,
    billedTokens: a.tokens,
    outputTokens: a.outputTokens ?? 0,
    inputTokens: inp,
    cachedInputTokens: cached,
  });
  const w = water(r.effectiveTokens, r.modelClass, a.provider, a.region);
  const wMl = w.available ? (w.mlTotalReig ?? null) : null;
  const wMlX = w.available ? (w.mlTotalExclHydro ?? null) : null;
  return {
    kgCo2e: r.kgCo2e,
    kgCo2eHighCase: r.kgCo2eHighCase,
    developmentLineKgCo2e: r.developmentLineKgCo2e,
    termsKgCo2e: {
      electricity: r.electricityKgCo2e,
      embodied: r.embodiedKgCo2e,
      construction: r.constructionKgCo2e,
      training: r.trainingKgCo2e,
    },
    kwh: r.gridKgCo2ePerKwh > 0 ? r.electricityKgCo2e / r.gridKgCo2ePerKwh : null,
    litres: wMl === null ? null : wMl / 1000,
    litresExclHydro: wMlX === null ? null : wMlX / 1000,
    waterMl: wMl,
    waterMlExclHydro: wMlX,
    model: a.model ?? null,
    modelClass: r.modelClass,
    classBasis: r.classBasis,
    region: r.regionId,
    gridKgCo2ePerKwh: r.gridKgCo2ePerKwh,
    billedTokens: r.billedTokens,
    effectiveTokens: r.effectiveTokens,
    warnings: [...new Set([...r.warnings, ...w.warnings])],
    factorVersion: DATASET_VERSION,
    note: NOTE,
  };
}

export interface Usage {
  model: string | null;
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
}

type Obj = Record<string, unknown>;
const get = (o: unknown, name: string): unknown =>
  o === null || o === undefined || typeof o !== "object" ? undefined : (o as Obj)[name];
const num = (o: unknown, name: string): number => {
  const v = get(o, name);
  return typeof v === "number" ? v : 0;
};
const has = (o: unknown, name: string): boolean => !isNil(get(o, name));

/**
 * Returns the uncached input, cached input and output tokens of an OpenAI Chat Completions, OpenAI Responses or
 * Anthropic Messages response. OpenAI counts cached tokens inside the prompt total. Anthropic reports cache reads
 * and cache writes beside input_tokens. Cache writes are treated as ordinary uncached input and only cache reads
 * get the cache credit.
 */
export function usageFromResponse(resp: unknown): Usage {
  const usage = get(resp, "usage");
  if (isNil(usage)) throw new SaefError("response has no usage block");
  const model = (get(resp, "model") as string | undefined) ?? null;
  if (has(usage, "prompt_tokens")) {
    const prompt = num(usage, "prompt_tokens");
    const cached = num(get(usage, "prompt_tokens_details"), "cached_tokens");
    return { model, inputTokens: Math.max(prompt - cached, 0), cachedInputTokens: cached, outputTokens: num(usage, "completion_tokens") };
  }
  if (!has(usage, "input_tokens")) throw new SaefError("usage has neither prompt_tokens nor input_tokens");
  const inp = num(usage, "input_tokens");
  const out = num(usage, "output_tokens");
  const details = get(usage, "input_tokens_details");
  if (!isNil(details) || get(resp, "object") === "response") {
    const cached = num(details, "cached_tokens");
    return { model, inputTokens: Math.max(inp - cached, 0), cachedInputTokens: cached, outputTokens: out };
  }
  const read = num(usage, "cache_read_input_tokens");
  const write = num(usage, "cache_creation_input_tokens");
  return { model, inputTokens: inp + write, cachedInputTokens: read, outputTokens: out };
}

export function defaultRegion(): string {
  const env = typeof process !== "undefined" ? process.env?.SAEF_REGION : undefined;
  return env || "global-average";
}

export interface ResponseOptions {
  region?: string | null;
  provider?: string | null;
  /** Used when the response carries no model. */
  model?: string;
}

/** Estimate for one OpenAI or Anthropic response object (or the same shape as a plain object). */
export function estimateFromResponse(resp: unknown, opts: ResponseOptions = {}): Estimate {
  const u = usageFromResponse(resp);
  const model = opts.model || u.model;
  if (!model) throw new SaefError("response has no model, pass model");
  return estimate({
    model,
    region: opts.region || defaultRegion(),
    provider: opts.provider,
    inputTokens: u.inputTokens,
    cachedInputTokens: u.cachedInputTokens,
    outputTokens: u.outputTokens,
  });
}

export interface TrackerOptions {
  region?: string | null;
  provider?: string | null;
}

export interface TrackerCall {
  model: string;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens?: number;
  region?: string | null;
  provider?: string | null;
}

export interface TrackerTotals {
  calls: number;
  skippedCalls: number;
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  kgCo2e: number;
  kgCo2eHighCase: number;
  kwh: number;
  litres: number;
  callsWithoutWaterFactor: number;
  byModel: Record<string, { calls: number; kgCo2e: number }>;
}

/** Running totals of per-call estimates. */
export class Tracker {
  readonly region: string;
  readonly provider: string | null;
  private t!: TrackerTotals;

  constructor(opts: TrackerOptions = {}) {
    this.region = opts.region || defaultRegion();
    this.provider = opts.provider ?? null;
    this.reset();
  }

  reset(): void {
    this.t = {
      calls: 0,
      skippedCalls: 0,
      inputTokens: 0,
      cachedInputTokens: 0,
      outputTokens: 0,
      kgCo2e: 0,
      kgCo2eHighCase: 0,
      kwh: 0,
      litres: 0,
      callsWithoutWaterFactor: 0,
      byModel: {},
    };
  }

  private run(c: TrackerCall): Estimate {
    const base = {
      model: c.model,
      provider: c.provider ?? this.provider,
      inputTokens: c.inputTokens,
      outputTokens: c.outputTokens,
      cachedInputTokens: c.cachedInputTokens ?? 0,
    };
    try {
      return estimate({ ...base, region: c.region || this.region });
    } catch (e) {
      if (e instanceof SaefError && c.region && c.region !== this.region) {
        return estimate({ ...base, region: this.region });
      }
      throw e;
    }
  }

  /**
   * Estimate one call, add it to the totals and return the estimate, or null if the model is out of scope.
   * A per-call region that SAEF does not know falls back to the tracker's region.
   */
  add(c: TrackerCall): Estimate | null {
    let e: Estimate;
    try {
      e = this.run(c);
    } catch (err) {
      if (err instanceof SaefError) {
        this.t.skippedCalls += 1;
        return null;
      }
      throw err;
    }
    const t = this.t;
    t.calls += 1;
    t.inputTokens += c.inputTokens;
    t.cachedInputTokens += c.cachedInputTokens ?? 0;
    t.outputTokens += c.outputTokens;
    t.kgCo2e += e.kgCo2e;
    t.kgCo2eHighCase += e.kgCo2eHighCase;
    t.kwh += e.kwh ?? 0;
    if (e.litres === null) t.callsWithoutWaterFactor += 1;
    else t.litres += e.litres;
    const m = (t.byModel[c.model] ??= { calls: 0, kgCo2e: 0 });
    m.calls += 1;
    m.kgCo2e += e.kgCo2e;
    return e;
  }

  /** Same as add(), reading the usage from an OpenAI or Anthropic response object. */
  addResponse(resp: unknown, opts: ResponseOptions = {}): Estimate | null {
    const u = usageFromResponse(resp);
    const model = opts.model || u.model;
    if (!model) throw new SaefError("response has no model, pass model");
    return this.add({
      model,
      inputTokens: u.inputTokens,
      cachedInputTokens: u.cachedInputTokens,
      outputTokens: u.outputTokens,
      region: opts.region,
      provider: opts.provider,
    });
  }

  totals(): TrackerTotals {
    return { ...this.t, byModel: Object.fromEntries(Object.entries(this.t.byModel).map(([k, v]) => [k, { ...v }])) };
  }
}

/** Cloud region ids (AWS, Azure, Google Cloud), sorted. Pass 'aws', 'azure' or 'gcp' to filter. */
export const regions = listRegions;

/** Grid ids usable as a region: ISO-3 country codes, eGRID subregions, EU27, WORLD and one subnational row. */
export const grids = listGrids;

/** Class (A, B or C) of a model identifier. Unmapped identifiers get the conservative class C. Throws for out-of-scope and non-text models. */
export function modelClassFor(model: string): string {
  const m = mapModel(model);
  if (m.outOfScope) throw new SaefError(`'${model}' is out of scope (text-to-speech, realtime voice, video and other types with no factor yet)`);
  if (m.modelType !== "text") throw new SaefError(`'${model}' has no text class: model type ${m.modelType}, class ${m.modelClass}`);
  return m.modelClass as string;
}

/** Kg CO2e per million billed tokens at the default input share, H100 central case. */
export function factor(modelClass = "B", region: string | null = "us-east-1", provider?: string | null) {
  const r = emissions({ modelClass, provider, region, billedTokens: 1e6 });
  const f = r.factors;
  return {
    modelClass: r.modelClass,
    region: r.regionId,
    gridId: r.gridId,
    gridKgCo2ePerKwh: r.gridKgCo2ePerKwh,
    kgCo2ePer1mBilledTokens: f.perBilled,
    kgCo2ePer1mEffectiveTokens: f.footprint,
    kgCo2ePer1mEffectiveTokensHighCase: f.footprintHigh,
    terms: {
      electricity: f.electricity,
      embodied: f.embodied,
      construction: f.construction,
      trainingPer1mBilledTokens: f.training,
    },
    effectivePerBilledTokens: effectiveFromBilled(1),
    version: DATASET_VERSION,
    warnings: r.warnings,
  };
}

/** BibTeX for the dataset (Zenodo concept DOI 10.5281/zenodo.20443585) and the method paper. */
export const citation = (): string => BIBTEX;
