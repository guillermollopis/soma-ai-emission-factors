import raw from "../data/saef-data.json";

export class SaefError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SaefError";
  }
}

type Row = Record<string, string>;
const D = raw as unknown as {
  datasetVersion: string;
  datasetDoi: string;
  datasetVersionDoi: string;
  paperDoi: string;
  constants: Row[];
  modelClassMapping: Row[];
  modelFactors: Row[];
  cloudRegions: Row[];
  grids: Row[];
  crosswalk: Row[];
  waterFactors: Row[];
  lifecycle: Row[];
};

export const DATASET_VERSION = D.datasetVersion;
export const DATASET_DOI = D.datasetDoi;
export const DATASET_VERSION_DOI = D.datasetVersionDoi;
export const PAPER_DOI = D.paperDoi;

const PATH_NOISE = new Set([
  "models", "model", "publishers", "registries", "versions", "version", "projects", "locations",
  "deployments", "inference-profile", "inference-profiles", "endpoints", "foundry", "openai",
  "azureml:", "azureml", "arn:aws:bedrock:", "providers", "catalogs", "assets",
]);
const DOT_REGION = new Set(["us", "eu", "apac", "us-gov", "global", "jp", "au", "ca", "sa", "me", "il", "uk", "in"]);
const DOT_VENDOR = new Set([
  "anthropic", "amazon", "meta", "mistral", "cohere", "ai21", "stability", "deepseek", "openai",
  "google", "writer", "luma", "twelvelabs", "qwen", "nvidia", "moonshot", "zai", "minimax",
  "microsoft", "xai", "alibaba", "tencent", "baidu", "perplexity", "snowflake", "databricks",
]);
const UNKNOWN_REGION = new Set(["", "unknown", "global", "global-average", "world", "any", "none", "mixed", "multi-region"]);
const EU27_REGION = new Set(["eu-27", "eu27", "eu", "eu-only", "eu data zone", "european union"]);

const pyRepr = (s: unknown): string => {
  const t = String(s);
  const q = t.includes("'") && !t.includes('"') ? '"' : "'";
  return q + t.split("\\").join("\\\\").split(q).join("\\" + q) + q;
};

const constants = new Map(D.constants.map((r) => [r.constant, r]));
const constant = (name: string): number => parseFloat(constants.get(name)!.central);

const CAMEL_BOUNDARY = /(?<=[a-z0-9])(?=[A-Z])/g;

export function normaliseIdentifier(identifier: string): string[] {
  const trimmed = String(identifier).trim();
  const spaced = trimmed.replace(CAMEL_BOUNDARY, "-");
  if (spaced !== trimmed) return [...new Set([...normaliseLower(trimmed), ...normaliseLower(spaced)])];
  return normaliseLower(trimmed);
}

function normaliseLower(identifier: string): string[] {
  const s = String(identifier).trim().toLowerCase();
  const cands = [s];
  let body = s;
  if (body.startsWith("arn:aws:bedrock:")) body = body.slice(body.lastIndexOf("/") + 1);
  const segs = body.split("/").filter((x) => x && !PATH_NOISE.has(x) && !/^[0-9]+$/.test(x));
  if (segs.length) body = segs[segs.length - 1];
  cands.push(body);
  let stripped = body;
  const dropped: string[] = [];
  while (stripped.includes(".")) {
    const i = stripped.indexOf(".");
    const head = stripped.slice(0, i);
    if (DOT_REGION.has(head) || DOT_VENDOR.has(head)) {
      dropped.push(head);
      stripped = stripped.slice(i + 1);
    } else break;
  }
  if (stripped !== body) {
    cands.push(stripped);
    if (dropped.length && DOT_VENDOR.has(dropped[dropped.length - 1])) {
      cands.push(`${dropped[dropped.length - 1]}-${stripped}`);
    }
  }
  const dotted = cands.filter((c) => c.includes(".")).map((c) => c.split(".").join("-"));
  return [...new Set([...cands, ...dotted].filter((c) => c))];
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

type Key = [number, number, number, number, string];
type Pattern = { key: Key; rx: RegExp; row: Row };
const cmpKey = (a: Key, b: Key): number => {
  for (let i = 0; i < 4; i++) if (a[i] !== b[i]) return (a[i] as number) - (b[i] as number);
  return a[4] < b[4] ? -1 : a[4] > b[4] ? 1 : 0;
};

let patternsCache: Pattern[] | null = null;
function patterns(): Pattern[] {
  if (patternsCache) return patternsCache;
  const out: Pattern[] = [];
  for (const row of D.modelClassMapping) {
    for (let alt of row.pattern.split(";")) {
      alt = alt.trim().toLowerCase();
      if (!alt) continue;
      const litPrefix = alt.split("*", 1)[0];
      const rx = new RegExp("^(?:" + alt.split("*").map(escapeRe).join(".*") + ")", "s");
      out.push({ key: [row.model_type !== "text" ? 1 : 0, litPrefix.length, alt.split("*").join("").length, alt.length, alt], rx, row });
    }
  }
  out.sort((a, b) => cmpKey(b.key, a.key));
  return (patternsCache = out);
}

let measuredCache: { byid: Map<string, string>; bylast: Map<string, string> } | null = null;
function measuredId(cands: string[]): string | null {
  if (!measuredCache) {
    const byid = new Map<string, string>();
    for (const r of D.modelFactors) byid.set(r.model_id.toLowerCase(), r.model_id);
    const bylast = new Map<string, string>();
    for (const [k, v] of byid) bylast.set(k.slice(k.lastIndexOf("/") + 1), v);
    measuredCache = { byid, bylast };
  }
  for (let c of cands) {
    c = c.trim().toLowerCase();
    if (measuredCache.byid.has(c)) return measuredCache.byid.get(c)!;
    if (measuredCache.bylast.has(c)) return measuredCache.bylast.get(c)!;
  }
  return null;
}

export interface ModelMapping {
  identifier: string;
  pattern: string;
  basis: string;
  modelClass: string | null;
  /** "text", "out-of-scope", or a non-text model type such as "embedding_text". */
  modelType: string;
  ownFactorModelId: string | null;
  outOfScope: boolean;
  isDefaultFallback: boolean;
  warnings: string[];
}

export function mapModel(identifier: string): ModelMapping {
  const cands = normaliseIdentifier(identifier);
  const hit = patterns().find((p) => cands.some((c) => p.rx.test(c)))?.row;
  if (!hit) throw new SaefError("model_class_mapping.csv has no default row");
  const warnings: string[] = [];
  const out: ModelMapping = {
    identifier: String(identifier),
    pattern: hit.pattern,
    basis: hit.basis,
    modelClass: hit.model_class,
    modelType: hit.model_type,
    ownFactorModelId: hit.own_factor_model_id || null,
    outOfScope: hit.model_class === "out-of-scope",
    isDefaultFallback: hit.basis === "default",
    warnings,
  };
  if (out.isDefaultFallback) {
    warnings.push(`unmapped model identifier ${pyRepr(identifier)}: defaulted to Class ${hit.model_class} and flagged`);
  }
  if (out.outOfScope) {
    out.modelClass = null;
    warnings.push(`${pyRepr(identifier)} matches the out-of-scope pattern ${pyRepr(hit.pattern)}: ${hit.notes}`);
  } else if (hit.model_type !== "text") {
    warnings.push(
      `${pyRepr(identifier)} is mapped to model type ${hit.model_type}, class ${hit.model_class}: an assignment by ` +
        `size, price tier or lineage, not a provider disclosure (${hit.size_or_tier_note})`,
    );
    return out;
  } else if (hit.basis === "tier") {
    warnings.push(
      `class proxy for a closed model: ${pyRepr(identifier)} was placed in Class ${hit.model_class} by the ` +
        `vendor pricing tier, not by a disclosed size (${hit.size_or_tier_note})`,
    );
  }
  const direct = measuredId([...cands, String(identifier)]);
  if (direct) {
    out.ownFactorModelId = direct;
    out.basis = "measured";
  }
  return out;
}

const levenshtein = (a: string, b: string): number => {
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
};

function suggest(region: string): string[] {
  const pool = [...new Set([...D.cloudRegions.map((r) => r.region_id), ...D.grids.map((r) => r.grid_id)])].sort();
  const low = region.toLowerCase();
  const near = pool.filter((x) => x.toLowerCase().startsWith(low)).slice(0, 8);
  if (near.length) return near;
  return pool
    .map((x) => ({ x, s: 1 - levenshtein(region, x) / Math.max(region.length, x.length, 1) }))
    .filter((m) => m.s >= 0.6)
    .sort((a, b) => b.s - a.s)
    .slice(0, 5)
    .map((m) => m.x);
}

interface Resolved {
  gridId: string;
  gridKgCo2ePerKwh: number;
  classRow: Row;
  regionId: string;
  how: string;
  warnings: string[];
}

export function resolveRegion(provider?: string | null, region?: string | null): Resolved {
  const warnings: string[] = [];
  const grids = new Map(D.grids.map((r) => [r.grid_id, r]));
  const key = (region || "").trim();
  const low = key.toLowerCase();

  const finish = (grid: Row, how: string, classRow?: Row, regionId?: string): Resolved => ({
    gridId: grid.grid_id,
    gridKgCo2ePerKwh: parseFloat(grid.grid_kg_co2e_per_kwh),
    classRow: classRow ?? grid,
    regionId: regionId || grid.grid_id,
    how,
    warnings,
  });

  if (UNKNOWN_REGION.has(low)) {
    warnings.push("region unknown or global: the global-average grid row (WORLD) is applied, per the dataset's serving-region decision rule");
    return finish(grids.get("WORLD")!, "unknown -> global average", undefined, "global-average");
  }
  if (EU27_REGION.has(low)) {
    warnings.push("EU data zone or EU-only processing: the EU-27 average grid row (EU27) is applied");
    return finish(grids.get("EU27")!, "EU-only processing -> EU-27", undefined, "eu-27");
  }
  let cloud = D.cloudRegions.filter((r) => r.region_id.toLowerCase() === low);
  if (provider) cloud = cloud.filter((r) => r.provider.toLowerCase() === provider.trim().toLowerCase());
  if (cloud.length) {
    if (new Set(cloud.map((r) => r.grid_id)).size > 1) {
      throw new SaefError(`Region ${pyRepr(region)} is used by several providers with different grids, pass provider`);
    }
    const r = cloud[0];
    if (r.mapping_confidence === "medium") {
      warnings.push(`region mapping confidence is medium for ${r.provider}/${r.region_id}: ${r.mapping_notes}`);
    }
    return finish(grids.get(r.grid_id)!, `cloud region ${r.provider}/${r.region_id}`, r, r.region_id);
  }
  const hit = D.crosswalk.find((r) => r.region_id.toLowerCase() === low);
  if (hit) return finish(grids.get(hit.grid_id)!, `printed region ${hit.region_id}`, undefined, hit.region_id);
  if (grids.has(key.toUpperCase())) {
    const g = grids.get(key.toUpperCase())!;
    return finish(g, `${g.kind.split("_").join(" ")} ${g.grid_id}`);
  }
  const byName = [...grids.values()].filter((r) => r.name.trim().toLowerCase() === low);
  if (byName.length === 1) return finish(byName[0], `country name -> ${byName[0].grid_id}`);
  const hint = suggest(key);
  let msg = `Unknown region ${pyRepr(region)}.`;
  if (hint.length) msg += ` Did you mean: ${hint.join(", ")}?`;
  msg += " Use a cloud region id (see regions()), an ISO-3 country code, an eGRID subregion, 'eu-27' or 'unknown'.";
  throw new SaefError(msg);
}

export const effectiveTokens = (output = 0, uncachedInput = 0, cachedInput = 0): number =>
  (output + constant("W_in") * uncachedInput + constant("W_cache") * cachedInput) / constant("D");

export const effectiveFromBilled = (billed: number): number => {
  const s = constant("default_input_share");
  return (billed * (1 - s + constant("W_in") * s)) / constant("D");
};

export interface Terms {
  footprint: number;
  training: number;
  footprintHigh: number;
  development: number;
  electricity: number;
  embodied: number;
  construction: number;
  perBilled: number;
  modelClass?: string;
  basis: string;
}

function classTerms(cl: string, reg: Resolved): Terms {
  const c = cl.toLowerCase();
  const row = reg.classRow;
  const col = (s: string) => `class_${c}_${s}`;
  if (!(col("inference_footprint_kg_per_m_effective_tokens") in row)) {
    throw new SaefError(`no class factors for model class ${pyRepr(cl)}; use A, B or C`);
  }
  const g = (s: string) => parseFloat(row[col(s)]);
  return {
    footprint: g("inference_footprint_kg_per_m_effective_tokens"),
    training: g("training_kg_per_m_billed_tokens"),
    footprintHigh: g("inference_footprint_high_case_kg_per_m_effective_tokens"),
    development: g("development_line_kg_per_m_effective_tokens"),
    electricity: g("electricity_kg_per_m_effective_tokens"),
    embodied: g("embodied_kg_per_m_effective_tokens"),
    construction: g("construction_kg_per_m_effective_tokens"),
    perBilled: g("factor_kg_per_m_billed_tokens_default_mix"),
    basis: `class ${cl} mean of the measured members, H100`,
  };
}

function modelTerms(modelId: string, reg: Resolved): Terms | null {
  const r = D.modelFactors.find((x) => x.model_id === modelId && x.gpu === "H100");
  if (!r) return null;
  const grid = reg.gridKgCo2ePerKwh;
  const f = parseFloat;
  const fac = f(r.facility_wh_per_1k_effective_tokens);
  const facHi = f(r.facility_wh_high_case_per_1k_effective_tokens);
  const emb = f(r.embodied_kg_co2e_per_m_tokens);
  const con = f(r.construction_kg_co2e_per_m_tokens);
  const embHi = f(r.embodied_high_case_kg_co2e_per_m_tokens);
  const conHi = f(r.construction_high_case_kg_co2e_per_m_tokens);
  const tr = f(r.training_kg_co2e_per_m_billed_tokens);
  const share = constant("default_input_share");
  const mult = (1 - share + constant("W_in") * share) / constant("D");
  const foot = fac * grid + emb + con;
  return {
    footprint: foot,
    training: tr,
    footprintHigh: facHi * grid + embHi + conHi,
    development: (f(r.development_line_kg_co2e_per_m_tokens_global) / f(r.grid_kg_co2e_per_kwh)) * grid,
    electricity: fac * grid,
    embodied: emb,
    construction: con,
    perBilled: foot * mult + tr,
    modelClass: r.model_class,
    basis: `measured model ${modelId}, H100`,
  };
}

export interface Emissions {
  kgCo2e: number;
  kgCo2eHighCase: number;
  developmentLineKgCo2e: number;
  electricityKgCo2e: number;
  embodiedKgCo2e: number;
  constructionKgCo2e: number;
  trainingKgCo2e: number;
  matchedPattern: string | null;
  modelClass: string;
  classBasis: string;
  ownFactorModelId: string | null;
  isDefaultFallback: boolean;
  regionId: string;
  gridId: string;
  gridKgCo2ePerKwh: number;
  regionResolution: string;
  billedTokens: number;
  effectiveTokens: number;
  factors: Terms;
  warnings: string[];
}

export interface EmissionsInput {
  model?: string | null;
  modelClass?: string | null;
  provider?: string | null;
  region?: string | null;
  billedTokens?: number | null;
  outputTokens?: number;
  inputTokens?: number;
  cachedInputTokens?: number;
}

export function emissions(a: EmissionsInput): Emissions {
  const warnings: string[] = [];
  const hasModel = a.model != null;
  if (hasModel === (a.modelClass != null)) throw new SaefError("Pass exactly one of model or modelClass");
  let cl: string;
  let own: string | null = null;
  let basis: string;
  let matched: string | null;
  let fallback = false;
  if (hasModel) {
    const mm = mapModel(a.model as string);
    warnings.push(...mm.warnings);
    if (mm.outOfScope) {
      throw new SaefError(
        `${pyRepr(a.model)} is out of scope (no model type covers it): no estimate. ${mm.warnings[mm.warnings.length - 1]}`,
      );
    }
    if (mm.modelType !== "text") {
      throw new SaefError(
        `${pyRepr(a.model)} has model type ${mm.modelType}, class ${mm.modelClass}: this package estimates text models only, ` +
          `use estimate_modality in the Python package`,
      );
    }
    cl = mm.modelClass as string;
    own = mm.ownFactorModelId;
    basis = mm.basis;
    matched = mm.pattern;
    fallback = mm.isDefaultFallback;
  } else {
    const letter = ({ a: "A", b: "B", c: "C" } as Record<string, string>)[
      String(a.modelClass).trim().toLowerCase().replace("class_", "").slice(0, 1)
    ];
    if (!letter) throw new SaefError(`Unknown model class ${pyRepr(a.modelClass)}; use A, B or C`);
    cl = letter;
    basis = "caller-supplied";
    matched = null;
  }
  const reg = resolveRegion(a.provider, a.region);
  warnings.push(...reg.warnings);

  let billed: number;
  let tEff: number;
  const cached = a.cachedInputTokens ?? 0;
  if (a.billedTokens != null) {
    billed = a.billedTokens;
    tEff = effectiveFromBilled(billed);
    warnings.push(
      `no input/output split given: the default input share ${constant("default_input_share")} and no cache credit were applied`,
    );
  } else {
    const out = a.outputTokens ?? 0;
    const inp = a.inputTokens ?? 0;
    billed = inp + cached + out;
    tEff = effectiveTokens(out, inp, cached);
    if (!cached) warnings.push("no cached-token count given, so no cache credit was taken");
  }

  let f = own ? modelTerms(own, reg) : null;
  if (f) {
    cl = f.modelClass as string;
    basis = "measured";
  } else {
    f = classTerms(cl, reg);
  }
  const m = tEff / 1e6;
  const b = billed / 1e6;
  return {
    kgCo2e: m * f.footprint + b * f.training,
    kgCo2eHighCase: m * f.footprintHigh + b * f.training,
    developmentLineKgCo2e: m * f.development,
    electricityKgCo2e: m * f.electricity,
    embodiedKgCo2e: m * f.embodied,
    constructionKgCo2e: m * f.construction,
    trainingKgCo2e: b * f.training,
    matchedPattern: matched,
    modelClass: cl,
    classBasis: basis,
    ownFactorModelId: own,
    isDefaultFallback: hasModel && fallback,
    regionId: reg.regionId,
    gridId: reg.gridId,
    gridKgCo2ePerKwh: reg.gridKgCo2ePerKwh,
    regionResolution: reg.how,
    billedTokens: billed,
    effectiveTokens: tEff,
    factors: f,
    warnings,
  };
}

export interface Water {
  available: boolean;
  regionId: string;
  mlOnsite?: number;
  mlOffsiteReig?: number | null;
  mlOffsiteExclHydro?: number | null;
  mlTotalReig?: number | null;
  mlTotalExclHydro?: number | null;
  warnings: string[];
}

export function water(tEff: number, modelClass: string, provider?: string | null, region?: string | null): Water {
  const warnings: string[] = [];
  const reg = resolveRegion(provider, region);
  warnings.push(...reg.warnings);
  const rid = reg.regionId;
  const wf = new Map(D.waterFactors.map((r) => [r.region_id, r]));
  if (reg.how.startsWith("unknown") || !wf.has(rid)) {
    warnings.push(
      `no water factor for region ${pyRepr(rid)}: region_water_factors.csv covers ${wf.size} regions only ` +
        `and has no unknown-region fallback. Disclose water as not available`,
    );
    return { available: false, regionId: rid, warnings };
  }
  const w = wf.get(rid)!;
  const fl = D.lifecycle.find((r) => r.model_class === modelClass && r.gpu === "H100" && r.region_id === rid);
  if (!fl) throw new SaefError(`no factors_v3_lifecycle.csv row for class ${modelClass} / H100 / ${rid}`);
  const fac = parseFloat(fl.facility_wh_per_1k_effective_tokens);
  const srv = parseFloat(fl.server_wh_per_1k_effective_tokens);
  const k = tEff / 1000;
  const base = w.wue_denominator.startsWith("IT load") ? srv : fac;
  let onsite = base * parseFloat(w.wue_l_per_kwh);
  if (w.wue_basis_as_published === "withdrawal") onsite *= parseFloat(w.withdrawal_to_consumption);
  const out: Water = {
    available: true,
    regionId: rid,
    mlOnsite: onsite * k,
    mlOffsiteReig: null,
    mlOffsiteExclHydro: null,
    mlTotalReig: null,
    mlTotalExclHydro: null,
    warnings,
  };
  if (!w.ewif_l_per_kwh) {
    warnings.push(`no off-site water factor for ${rid}: ${w.ewif_source}. Only the on-site term is reported`);
  } else {
    const off = fac * parseFloat(w.ewif_l_per_kwh);
    const offX = fac * parseFloat(w.ewif_excl_hydro_l_per_kwh);
    out.mlOffsiteReig = off * k;
    out.mlOffsiteExclHydro = offX * k;
    out.mlTotalReig = (onsite + off) * k;
    out.mlTotalExclHydro = (onsite + offX) * k;
    warnings.push(
      "two hydropower conventions are reported and neither is preferred: 'reig' allocates reservoir evaporation to electricity, 'excl_hydro' excludes it",
    );
  }
  return out;
}

export const listRegions = (provider?: string): string[] =>
  [
    ...new Set(
      D.cloudRegions
        .filter((r) => !provider || r.provider.toLowerCase() === provider.trim().toLowerCase())
        .map((r) => r.region_id),
    ),
  ].sort();

export const listGrids = (): string[] => D.grids.map((r) => r.grid_id).sort();
