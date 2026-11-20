import type { LanguageModelMiddleware } from "ai";
import { SaefError, Tracker, estimate, defaultRegion, type Estimate } from "./index.js";

export interface SaefMiddlewareContext {
  modelId: string;
  provider: string | undefined;
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
}

export interface SaefMiddlewareOptions {
  /** Cloud region id, ISO-3 country code, eGRID subregion or 'eu-27'. Defaults to SAEF_REGION, then the global average. */
  region?: string;
  /** Cloud provider ('aws', 'azure', 'gcp') for region ids shared by several providers. */
  cloudProvider?: string;
  /** Called with one estimate per finished model call. */
  onEstimate?: (estimate: Estimate, ctx: SaefMiddlewareContext) => void;
  /** Called when a call cannot be estimated (for example an out-of-scope model). The model call itself is never affected. */
  onError?: (error: unknown, ctx: { modelId: string | undefined; provider: string | undefined }) => void;
  /** Running totals. Each estimated call is added to it. */
  tracker?: Tracker;
}

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj | undefined => (v !== null && typeof v === "object" ? (v as Obj) : undefined);
const n = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

/**
 * Reads token usage from any AI SDK language-model spec version.
 * V2 and older, flat: inputTokens, outputTokens, cachedInputTokens, or promptTokens and completionTokens.
 * V3 and later, nested: inputTokens {total, noCache, cacheRead, cacheWrite}, outputTokens {total}.
 * Input tokens include cache reads, so the uncached input is the total minus cache reads. Cache writes count as
 * ordinary input, as in the Python package.
 */
export function readUsage(usage: unknown): { inputTokens: number; cachedInputTokens: number; outputTokens: number } | undefined {
  const u = obj(usage);
  if (!u) return undefined;
  const inp = obj(u.inputTokens);
  const out = obj(u.outputTokens);
  let total: number | undefined;
  let cached: number;
  if (inp) {
    cached = n(inp.cacheRead) ?? 0;
    total = n(inp.total) ?? ((n(inp.noCache) ?? 0) + cached + (n(inp.cacheWrite) ?? 0));
  } else {
    total = n(u.inputTokens) ?? n(u.promptTokens);
    cached = n(u.cachedInputTokens) ?? 0;
  }
  const output = (out ? (n(out.total) ?? n(out.text)) : (n(u.outputTokens) ?? n(u.completionTokens))) ?? undefined;
  if (total === undefined && output === undefined) return undefined;
  return {
    inputTokens: Math.max((total ?? 0) - cached, 0),
    cachedInputTokens: cached,
    outputTokens: output ?? 0,
  };
}

/**
 * Vercel AI SDK language model middleware that estimates the carbon, energy and water of each call.
 *
 * ```ts
 * const model = wrapLanguageModel({ model: openai("gpt-4o"), middleware: saefMiddleware({ region: "eu-west-1" }) });
 * ```
 */
export function saefMiddleware(options: SaefMiddlewareOptions = {}): LanguageModelMiddleware {
  const record = (model: { modelId?: string; provider?: string }, usage: unknown): void => {
    const ctx = { modelId: model.modelId, provider: model.provider };
    try {
      const u = readUsage(usage);
      if (!u) throw new SaefError("result has no token usage");
      if (!model.modelId) throw new SaefError("model has no modelId");
      const base = { ...u, modelId: model.modelId, provider: model.provider };
      let e: Estimate | null;
      if (options.tracker) {
        e = options.tracker.add({
          model: model.modelId,
          inputTokens: u.inputTokens,
          cachedInputTokens: u.cachedInputTokens,
          outputTokens: u.outputTokens,
          region: options.region,
          provider: options.cloudProvider,
        });
        if (!e) throw new SaefError("model out of scope or region unknown, call skipped");
      } else {
        e = estimate({
          model: model.modelId,
          inputTokens: u.inputTokens,
          cachedInputTokens: u.cachedInputTokens,
          outputTokens: u.outputTokens,
          region: options.region || defaultRegion(),
          provider: options.cloudProvider,
        });
      }
      options.onEstimate?.(e, base);
    } catch (error) {
      options.onError?.(error, ctx);
    }
  };

  return {
    wrapGenerate: async ({ doGenerate, model }) => {
      const result = await doGenerate();
      record(model, (result as unknown as Obj).usage);
      return result;
    },
    wrapStream: async ({ doStream, model }) => {
      const { stream, ...rest } = await doStream();
      const out = stream.pipeThrough(
        new TransformStream({
          transform(part, controller) {
            if (part.type === "finish") record(model, (part as unknown as Obj).usage);
            controller.enqueue(part);
          },
        }),
      );
      return { stream: out, ...rest };
    },
  };
}

export type { Estimate };
