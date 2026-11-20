import { describe, expect, it } from "vitest";
import { wrapLanguageModel } from "ai";
import type { LanguageModelV4 } from "@ai-sdk/provider";
import { Tracker, estimate } from "../src/index.js";
import { readUsage, saefMiddleware } from "../src/ai-sdk.js";

const usageV4 = {
  inputTokens: { total: 2000, noCache: 200, cacheRead: 1800, cacheWrite: 0 },
  outputTokens: { total: 150, text: 150, reasoning: undefined },
};
const expected = estimate({ model: "claude-sonnet-4-5", inputTokens: 200, cachedInputTokens: 1800, outputTokens: 150, region: "eu-west-1" });

const mock = (modelId = "claude-sonnet-4-5"): LanguageModelV4 => ({
  specificationVersion: "v4",
  provider: "anthropic.messages",
  modelId,
  supportedUrls: {},
  doGenerate: async () => ({
    content: [{ type: "text", text: "hi" }],
    finishReason: { unified: "stop", raw: "end_turn" },
    usage: usageV4,
    warnings: [],
  }),
  doStream: async () => ({
    stream: new ReadableStream({
      start(c) {
        c.enqueue({ type: "text-start", id: "1" });
        c.enqueue({ type: "text-delta", id: "1", delta: "hi" });
        c.enqueue({ type: "text-end", id: "1" });
        c.enqueue({ type: "finish", finishReason: { unified: "stop", raw: "end_turn" }, usage: usageV4 });
        c.close();
      },
    }),
  }),
}) as LanguageModelV4;

const params = { prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }] } as never;

describe("saefMiddleware", () => {
  it("wrapGenerate estimates from the result usage and returns the result untouched", async () => {
    const seen: unknown[] = [];
    const tracker = new Tracker({ region: "eu-west-1" });
    const model = wrapLanguageModel({
      model: mock(),
      middleware: saefMiddleware({ region: "eu-west-1", tracker, onEstimate: (e, ctx) => seen.push([e, ctx]) }),
    });
    const res = await model.doGenerate(params);
    expect(res.content[0]).toMatchObject({ type: "text", text: "hi" });
    expect(seen).toHaveLength(1);
    const [e, ctx] = seen[0] as [typeof expected, Record<string, unknown>];
    expect(e.kgCo2e).toBe(expected.kgCo2e);
    expect(e.litres).toBe(expected.litres);
    expect(ctx).toMatchObject({ modelId: "claude-sonnet-4-5", provider: "anthropic.messages", inputTokens: 200, cachedInputTokens: 1800, outputTokens: 150 });
    expect(tracker.totals().kgCo2e).toBe(expected.kgCo2e);
    expect(tracker.totals().calls).toBe(1);
  });

  it("wrapStream estimates on the finish part and passes every part through", async () => {
    const seen: number[] = [];
    const model = wrapLanguageModel({
      model: mock(),
      middleware: saefMiddleware({ region: "eu-west-1", onEstimate: (e) => seen.push(e.kgCo2e) }),
    });
    const { stream } = await model.doStream(params);
    const parts: { type: string }[] = [];
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value as { type: string });
    }
    expect(parts.map((p) => p.type)).toEqual(["text-start", "text-delta", "text-end", "finish"]);
    expect(seen).toEqual([expected.kgCo2e]);
  });

  it("never breaks the call: out-of-scope models go to onError", async () => {
    const errors: unknown[] = [];
    const model = wrapLanguageModel({
      model: mock("text-embedding-3-small"),
      middleware: saefMiddleware({ onError: (e) => errors.push(e) }),
    });
    const res = await model.doGenerate(params);
    expect(res.content).toHaveLength(1);
    expect(errors).toHaveLength(1);
  });

  it("defaults to the global average when no region is given", async () => {
    let region = "";
    const model = wrapLanguageModel({ model: mock(), middleware: saefMiddleware({ onEstimate: (e) => (region = e.region) }) });
    await model.doGenerate(params);
    expect(region).toBe("global-average");
  });
});

describe("readUsage", () => {
  it("reads V2 flat, older prompt/completion and V4 nested shapes", () => {
    expect(readUsage({ inputTokens: 2000, outputTokens: 150, cachedInputTokens: 1800 })).toEqual({ inputTokens: 200, cachedInputTokens: 1800, outputTokens: 150 });
    expect(readUsage({ inputTokens: 10, outputTokens: 5 })).toEqual({ inputTokens: 10, cachedInputTokens: 0, outputTokens: 5 });
    expect(readUsage({ promptTokens: 10, completionTokens: 5 })).toEqual({ inputTokens: 10, cachedInputTokens: 0, outputTokens: 5 });
    expect(readUsage(usageV4)).toEqual({ inputTokens: 200, cachedInputTokens: 1800, outputTokens: 150 });
    expect(readUsage({ inputTokens: { total: undefined, noCache: 7, cacheRead: 3, cacheWrite: 2 }, outputTokens: { total: 1 } })).toEqual({
      inputTokens: 9, cachedInputTokens: 3, outputTokens: 1,
    });
    expect(readUsage({ inputTokens: undefined, outputTokens: undefined })).toBeUndefined();
    expect(readUsage(undefined)).toBeUndefined();
  });
});
