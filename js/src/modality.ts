import {
  DATASET_VERSION,
  SaefError,
  mapModel,
  modalityFactorRows,
  pyRepr,
  resolveRegion,
  unitOnlyRow,
  waterForEnergy,
  type ModelMapping,
} from "./calc.js";

type Row = Record<string, string>;

const CASES = ["central", "low", "high"] as const;
const UNITS: Record<string, [string, number]> = {
  input_tokens: ["1k_input_tokens", 1e-3],
  cached_input_tokens: ["1k_cached_input_tokens", 1e-3],
  text_output_tokens: ["1k_text_output_tokens", 1e-3],
  image_output_tokens: ["1k_image_output_tokens", 1e-3],
  images: ["image", 1.0],
  images_1024_low: ["image_1024_low", 1.0],
  images_1024_medium: ["image_1024_medium", 1.0],
  images_1024_high: ["image_1024_high", 1.0],
  images_4k: ["image_4k", 1.0],
  search_units: ["search_unit", 1.0],
  calls: ["call", 1.0],
  audio_minutes: ["audio_minute", 1.0],
  audio_seconds: ["audio_minute", 1 / 60],
  characters: ["1k_characters", 1e-3],
  audio_output_tokens: ["1k_audio_output_tokens", 1e-3],
  user_audio_minutes: ["user_audio_minute", 1.0],
  assistant_audio_minutes: ["assistant_audio_minute", 1.0],
  context_minutes: ["context_minute", 1.0],
  conversation_minutes: ["conversation_minute", 1.0],
  active_session_minutes: ["active_session_minute", 1.0],
  video_seconds: ["video_second_720p", 1.0],
  video_seconds_480p: ["video_second_480p", 1.0],
  video_seconds_720p: ["video_second_720p", 1.0],
  video_seconds_1080p: ["video_second_1080p", 1.0],
  video_seconds_2160p: ["video_second_2160p", 1.0],
  video_clips: ["video_clip", 1.0],
};
const VIDEO_UNITS = Object.keys(UNITS).filter((u) => u.startsWith("video_"));
const VIDEO_INDICATIVE =
  "video values are indicative: the serving factor (0.5, range 0.2 to 1.0) and the resolution multipliers are " +
  "assumptions, commercial models are classed by price tier only, and ranges above 720p span 50 to 80 times";
const VOICE_MINUTES = ["user_audio_minutes", "assistant_audio_minutes", "context_minutes", "conversation_minutes"];
const ALIASES: Record<string, string> = {
  embedding_tokens: "input_tokens",
  tokens_scored: "input_tokens",
  audio_input_tokens: "input_tokens",
  text_input_tokens: "input_tokens",
  image_input_tokens: "input_tokens",
};
const UNIT_DEFAULTS: Record<string, [string, string]> = {
  image_output_tokens: ["image_generation_token_billed", "IMG-TOK-C"],
  images: ["image_generation_diffusion", "D-large"],
  images_1024_low: ["image_generation_token_billed", "IMG-TOK-C"],
  images_1024_medium: ["image_generation_token_billed", "IMG-TOK-C"],
  images_1024_high: ["image_generation_token_billed", "IMG-TOK-C"],
  images_4k: ["image_generation_token_billed", "IMG-TOK-C"],
  search_units: ["rerank", "R-large"],
  calls: ["moderation", "M-B"],
  audio_minutes: ["speech_to_text", "STT-LLM-B"],
  audio_seconds: ["speech_to_text", "STT-LLM-B"],
  characters: ["tts", "TTS-B"],
  audio_output_tokens: ["realtime_turn_based", "RT-C"],
  active_session_minutes: ["realtime_full_duplex", "FD-C"],
  ...Object.fromEntries(VOICE_MINUTES.map((u) => [u, ["realtime_turn_based", "RT-C"]])),
  ...Object.fromEntries(VIDEO_UNITS.map((u) => [u, ["video_generation", "V-C"]])),
};
const LOOSE_IMAGE_UNITS = ["images", "images_1024_low", "images_1024_medium", "images_1024_high", "images_4k"];
const MEGAPIXELS_1024 = 1.048576;
const NOTE =
  "Estimate, not a measurement. low and high are the class range of the GPU energy at the central overhead; the high " +
  "case is the central energy at the conservative overhead. No training term is charged for this model type.";

const VALUE_COLUMNS = [
  ...["gpu_wh", "server_wh", "facility_wh", "embodied_g_co2e", "construction_g_co2e"].flatMap((q) =>
    CASES.map((k) => `${q}_${k}`),
  ),
  "facility_wh_high_case",
  "embodied_g_co2e_high_case",
  "construction_g_co2e_high_case",
];

const camel = (s: string): string => s.replace(/_([a-z0-9])/g, (_m, c: string) => c.toUpperCase());

let rowsCache: Map<string, Row> | null = null;
const rowKey = (t: string, c: string, u: string) => `${t}\u0000${c}\u0000${u}`;
function rows(): Map<string, Row> {
  if (!rowsCache) {
    rowsCache = new Map(modalityFactorRows().map((r) => [rowKey(r.model_type, r.class, r.functional_unit), r]));
  }
  return rowsCache;
}

let classesCache: Map<string, Map<string, string>> | null = null;
function classes(): Map<string, Map<string, string>> {
  if (!classesCache) {
    classesCache = new Map();
    for (const r of modalityFactorRows()) {
      if (!classesCache.has(r.model_type)) classesCache.set(r.model_type, new Map());
      classesCache.get(r.model_type)!.set(r.class.toLowerCase(), r.class);
    }
  }
  return classesCache;
}

function canonicalUnit(unit: string): [string, number] {
  let u = String(unit).trim().toLowerCase().split("-").join("_").split(" ").join("_");
  u = ALIASES[u] ?? u;
  for (const name of [u, u + "s"]) {
    if (Object.prototype.hasOwnProperty.call(UNITS, name)) return [name, 1.0];
  }
  for (const [name, [functional, scale]] of Object.entries(UNITS)) {
    if (u === functional || u === "1k_" + name) return [name, 1 / scale];
  }
  throw new SaefError(`unknown usage unit ${pyRepr(unit)}; use one of ${Object.keys(UNITS).sort().join(", ")}`);
}

const pick = (types: string[], cls: string, unit: string): Row | null => {
  for (const t of types) {
    const r = rows().get(rowKey(t, cls, UNITS[unit][0]));
    if (r) return r;
  }
  return null;
};

type Mapping = Pick<ModelMapping, "modelType" | "modelClass" | "basis" | "outOfScope" | "isDefaultFallback" | "warnings">;

function mapping(model: string | null | undefined, modelType?: string | null, modelClass?: string | null): Mapping {
  if (model != null) {
    if (modelType != null || modelClass != null) throw new SaefError("Pass model, or model_type with model_class, not both");
    return mapModel(model);
  }
  const cls = classes().get(modelType as string)?.get(String(modelClass ?? "None").trim().toLowerCase());
  if (cls === undefined) {
    throw new SaefError(`no modality factors for model_type ${pyRepr(modelType)} class ${pyRepr(modelClass)}`);
  }
  return { modelType: modelType as string, modelClass: cls, basis: "caller-supplied", outOfScope: false, isDefaultFallback: false, warnings: [] };
}

function resolve(model: string | null | undefined, mm: Mapping, unit: string, warnings: string[]): Row {
  if (mm.outOfScope) {
    throw new SaefError(`${pyRepr(model)} is out of scope: no model type covers it yet. ${mm.warnings[mm.warnings.length - 1]}`);
  }
  const unitRow = model != null ? unitOnlyRow(model) : null;
  if (unitRow) {
    const row = pick([unitRow.model_type], unitRow.model_class, unit);
    if (row) {
      warnings.push(`${pyRepr(model)} is billed in ${unit}: routed by the billing unit to ${row.model_type}, class ${row.class}`);
      return row;
    }
  }
  const types = mm.modelType.split(";");
  if (mm.modelType !== "text") {
    const row = pick(types, mm.modelClass as string, unit);
    if (row) return row;
    const supported = new Set<string>();
    for (const r of modalityFactorRows()) if (r.functional_unit === UNITS[unit][0]) supported.add(r.model_type);
    if (!(unit in UNIT_DEFAULTS) || types.some((t) => supported.has(t)) || model == null) {
      const have = Object.keys(UNITS)
        .filter((n) => pick(types, mm.modelClass as string, n) !== null)
        .sort();
      throw new SaefError(
        `${pyRepr(model || mm.modelType)} (model type ${mm.modelType}, class ${mm.modelClass}) is not charged ` +
          `on ${unit}; pass one of ${have.join(", ")}`,
      );
    }
  } else if (!(unit in UNIT_DEFAULTS)) {
    throw new SaefError(`${pyRepr(model)} maps to the text factors; use estimate() for ${unit}`);
  }
  const [t, cls] = UNIT_DEFAULTS[unit];
  warnings.push(
    `${pyRepr(model)} does not map to a model type billed in ${unit}: routed by the billing unit to ${t}, ` +
      `default class ${cls}, and flagged`,
  );
  return pick([t], cls, unit) as Row;
}

type Component = Record<string, number | string | boolean>;

function component(row: Row, unit: string, count: number, n: number): Component {
  const out: Component = {};
  for (const c of VALUE_COLUMNS) out[camel(c)] = parseFloat(row[c]) * n;
  out.modelType = row.model_type;
  out.unit = unit;
  out.count = count;
  out.functionalUnit = row.functional_unit;
  out.basis = row.basis;
  out.sourceIds = row.source_ids;
  out.modelClass = row.class;
  return out;
}

function imageTokens(row: Row, tokens: number, images: number | null, warnings: string[]): Component {
  const fu = images === null ? "1k_image_output_tokens" : "1k_image_output_tokens_parity";
  const byTokens = component(rows().get(rowKey(row.model_type, row.class, fu)) as Row, "image_output_tokens", tokens, tokens / 1e3);
  if (images === null) {
    warnings.push(
      "image count unknown: the central value assumes 1,056 image tokens per image for the diffusion floor and " +
        "charges at least one image; pass images for a tighter value",
    );
    images = tokens ? 1.0 : 0.0;
  }
  const floorRow = rows().get(rowKey("image_generation_diffusion", "D-small", "image")) as Row;
  const floor = component(floorRow, "images", images, images);
  const out = { ...byTokens };
  for (const k of [...CASES, "high_case"]) {
    const src = k === "high_case" ? "central" : k;
    if ((floor[camel(`gpu_wh_${src}`)] as number) > (byTokens[camel(`gpu_wh_${src}`)] as number)) {
      for (const c of VALUE_COLUMNS) if (c.endsWith("_" + k)) out[camel(c)] = floor[camel(c)];
      if (k === "central") out.flooredToDiffusion = true;
    }
  }
  return out;
}

export interface ModalityOptions {
  /** Cloud region id, ISO-3 country code, eGRID subregion, 'eu-27', or omit for the global average. */
  region?: string | null;
  /** Cloud provider ('aws', 'azure', 'gcp') to disambiguate a region id shared by several providers. */
  provider?: string | null;
  /** Output size for per-image diffusion models above 1024x1024. */
  megapixels?: number | null;
  /** With modelClass, instead of a model identifier. */
  modelType?: string | null;
  modelClass?: string | null;
}

export interface ModalityEstimate {
  kgCo2e: number;
  kgCo2eLow: number;
  kgCo2eHigh: number;
  kgCo2eHighCase: number;
  termsKgCo2e: { electricity: number; embodied: number; construction: number; training: number };
  energyWh: number;
  energyWhLow: number;
  energyWhHigh: number;
  gpuEnergyWh: number;
  serverEnergyWh: number;
  /** Water in mL under the hydropower-allocated (reig) convention, null where the region has no water factor. */
  waterMl: number | null;
  waterMlLow: number | null;
  waterMlHigh: number | null;
  waterMlExclHydro: number | null;
  model: string | null;
  modelType: string;
  modelClass: string;
  classBasis: string;
  region: string;
  gridKgCo2ePerKwh: number;
  usage: Record<string, number>;
  components: Component[];
  warnings: string[];
  /** Sum of the token-unit counts, null when no token unit was charged. */
  billedTokens: number | null;
  factorVersion: string;
  note: string;
}

export type ModalityUsage = Record<string, number | null | undefined>;

/** Estimate for several billing units of one call, for example { input_tokens: 1200, images: 2 }. */
export function estimateModalityUsage(usage: ModalityUsage, model?: string | null, opts: ModalityOptions = {}): ModalityEstimate {
  const { region, provider, megapixels, modelType, modelClass } = opts;
  const counts: Record<string, number> = {};
  for (const [key, raw] of Object.entries(usage ?? {})) {
    if (raw === null || raw === undefined) continue;
    const [unit, scale] = canonicalUnit(key);
    const v = Number(raw) * scale;
    if (!(v >= 0)) throw new SaefError("usage counts must be >= 0");
    if (v === Infinity) throw new SaefError("usage counts must be finite");
    counts[unit] = (counts[unit] ?? 0) + v;
  }
  if (!Object.keys(counts).length) throw new SaefError("usage is empty; pass at least one unit and its count");
  const mm = mapping(model, modelType, modelClass);
  const warnings: string[] = [];
  const reg = resolveRegion(provider, region);
  warnings.push(...reg.warnings);

  if ("conversation_minutes" in counts && Object.keys(counts).length > 1) {
    throw new SaefError("conversation_minutes already covers the whole call; do not combine it with other units");
  }
  let floorImages: number | null = null;
  if ("image_output_tokens" in counts) {
    if (LOOSE_IMAGE_UNITS.slice(1).some((u) => u in counts)) {
      throw new SaefError("image_output_tokens and a per-image quality unit would count the same images twice; pass one");
    }
    if ("images" in counts) {
      floorImages = counts.images;
      delete counts.images;
    }
  }

  const components: Component[] = [];
  for (const [unit, count] of Object.entries(counts)) {
    const row = resolve(model, mm, unit, warnings);
    let n = count * UNITS[unit][1];
    if (unit === "image_output_tokens") {
      components.push(imageTokens(row, count, floorImages, warnings));
      continue;
    }
    if (megapixels != null && row.model_type === "image_generation_diffusion" && unit === "images") {
      if (!(Number(megapixels) > 0)) throw new SaefError("megapixels must be > 0");
      n *= Math.max(1.0, Number(megapixels) / MEGAPIXELS_1024);
    } else if (megapixels != null) {
      warnings.push("megapixels applies to per-image diffusion models only and was ignored");
    }
    components.push(component(row, unit, count, n));
  }

  if (components.some((c) => c.modelType === "video_generation")) warnings.push(VIDEO_INDICATIVE);
  const tot = (key: string): number => components.reduce((a, c) => a + (c[camel(key)] as number), 0);
  const server = tot("server_wh_central");
  const fac = tot("facility_wh_central");
  const grid = reg.gridKgCo2ePerKwh;
  const kg = {} as Record<(typeof CASES)[number], number>;
  for (const k of CASES) {
    kg[k] = (tot(`facility_wh_${k}`) / 1e3) * grid + (tot(`embodied_g_co2e_${k}`) + tot(`construction_g_co2e_${k}`)) / 1e3;
  }
  const highCase =
    (tot("facility_wh_high_case") / 1e3) * grid +
    (tot("embodied_g_co2e_high_case") + tot("construction_g_co2e_high_case")) / 1e3;
  const water = {} as Record<(typeof CASES)[number], ReturnType<typeof waterForEnergy>>;
  for (const k of CASES) {
    water[k] = waterForEnergy(() => [tot(`server_wh_${k}`), tot(`facility_wh_${k}`)], 1.0, provider, region);
  }
  const w = water.central;
  const ml = (k: (typeof CASES)[number]): number | null => (w.available ? (water[k].mlTotalReig ?? null) : null);
  const types = [...new Set(components.map((c) => c.modelType as string))];
  const cls = [...new Set(components.map((c) => c.modelClass as string))];
  let basis: string;
  if (mm.modelType !== "text" && types.some((t) => mm.modelType.split(";").includes(t))) {
    warnings.push(...mm.warnings);
    basis = mm.basis;
  } else {
    if (mm.isDefaultFallback) warnings.push(...mm.warnings);
    basis = "billing unit";
  }
  const tokenUnits = components.filter((c) => (c.unit as string).endsWith("_tokens")).map((c) => c.count as number);
  const usageOut: Record<string, number> = { ...counts };
  if (floorImages !== null) usageOut.images = floorImages;
  return {
    kgCo2e: kg.central,
    kgCo2eLow: kg.low,
    kgCo2eHigh: kg.high,
    kgCo2eHighCase: highCase,
    termsKgCo2e: {
      electricity: (fac / 1e3) * grid,
      embodied: tot("embodied_g_co2e_central") / 1e3,
      construction: tot("construction_g_co2e_central") / 1e3,
      training: 0.0,
    },
    energyWh: fac,
    energyWhLow: tot("facility_wh_low"),
    energyWhHigh: tot("facility_wh_high"),
    gpuEnergyWh: tot("gpu_wh_central"),
    serverEnergyWh: server,
    waterMl: ml("central"),
    waterMlLow: ml("low"),
    waterMlHigh: ml("high"),
    waterMlExclHydro: w.available ? (w.mlTotalExclHydro ?? null) : null,
    model: model ?? null,
    modelType: types.join(";"),
    modelClass: cls.join(";"),
    classBasis: basis,
    region: reg.regionId,
    gridKgCo2ePerKwh: grid,
    usage: usageOut,
    components,
    warnings: [...new Set([...warnings, ...w.warnings])],
    billedTokens: tokenUnits.length ? tokenUnits.reduce((a, b) => a + b, 0) : null,
    factorVersion: DATASET_VERSION,
    note: NOTE,
  };
}

/**
 * Estimate energy, kg CO2e and water for a non-text usage (embeddings, reranking, moderation, image generation,
 * transcription, text-to-speech, realtime voice, video), each with low, central and high. unit is the billed unit:
 * input_tokens, image_output_tokens, images, images_1024_high, audio_minutes, audio_seconds, search_units, calls,
 * characters, audio_output_tokens, conversation_minutes, active_session_minutes, video_seconds, video_seconds_1080p
 * and so on.
 */
export function estimateModality(
  unitCount: number,
  unit: string,
  model?: string | null,
  opts: ModalityOptions = {},
): ModalityEstimate {
  if (unit == null || unitCount == null) throw new SaefError("Pass unit_count and unit");
  return estimateModalityUsage({ [unit]: unitCount }, model, opts);
}
