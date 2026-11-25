# saef

JavaScript and TypeScript access to SAEF, the SOMA AI Emission Factors. SAEF is a versioned dataset of estimated carbon and water factors for AI inference, by model class and region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. This package bundles the factor tables and gives you a small API on top of them. It is a port of the [Python package](https://pypi.org/project/saef/) and gives the same numbers. It has no runtime dependencies.

Use it to estimate the carbon (kg CO2e), energy (kWh) and water (L) of OpenAI, Anthropic and other LLM API calls from the token usage each response already reports, with one function call on a response object or a Vercel AI SDK middleware. The method is published in Sustainable Production and Consumption 70 (2026) 320-347 (https://doi.org/10.1016/j.spc.2026.11.003).

## Install

```bash
npm install saef
```

Node 18 or later. ESM and CommonJS are both supported, with types. The Vercel AI SDK middleware needs the `ai` package, which is an optional peer dependency.

## Example

```ts
import { estimate, modelClassFor, factor, citation } from "saef";

console.log(modelClassFor("gpt-4o")); // "B"

const e = estimate({ model: "gpt-4o-mini", inputTokens: 1200, outputTokens: 300, region: "eu-west-1" });
console.log(e.kgCo2e, e.kgCo2eHighCase); // central and conservative high case, kg CO2e
console.log(e.kwh);                      // facility energy, kWh
console.log(e.litres);                   // water in litres, or null where the region has no water factor
console.log(e.modelClass, e.region, e.factorVersion); // "A" "eu-west-1" "3.4.0"

estimate({ model: "gpt-4o", tokens: 1_000_000, region: "eu-west-1" }); // a billed total with no split
console.log(factor("B", "eu-west-1").kgCo2ePer1mBilledTokens);
console.log(citation());
```

`inputTokens` is the uncached input. Pass cache reads separately as `cachedInputTokens`. Pass `tokens` (the billed total) instead of `outputTokens` when your invoice gives no split, which applies the default input share of 85% and no cache credit.

## Track OpenAI and Anthropic API calls

```ts
import { estimateFromResponse } from "saef";

const resp = await openai.chat.completions.create({ model: "gpt-4o-mini", messages });
const e = estimateFromResponse(resp, { region: "eu-west-1" });
console.log(e.kgCo2e, e.kgCo2eHighCase, e.litres);
```

No SDK import is needed, so the same call works on OpenAI Chat Completions, OpenAI Responses and Anthropic Messages objects, or the same shapes as plain objects. OpenAI counts cached tokens inside the prompt total. Anthropic reports cache reads and cache writes beside `input_tokens`. Cache reads get the cache credit and cache writes count as ordinary input.

```ts
const msg = await anthropic.messages.create({ model: "claude-sonnet-4-5", max_tokens: 512, messages });
const e = estimateFromResponse(msg, { region: "eu-west-1" });
```

Running totals per model:

```ts
import { Tracker } from "saef";

const tracker = new Tracker({ region: "eu-west-1" });
tracker.addResponse(resp);
tracker.add({ model: "gpt-4o", inputTokens: 900, cachedInputTokens: 300, outputTokens: 200 });
const t = tracker.totals();
console.log(t.calls, t.kgCo2e, t.kgCo2eHighCase, t.kwh, t.litres, t.byModel);
tracker.reset();
```

Non-text models, such as embeddings, images and audio, are skipped and counted in `totals().skippedCalls`. A per-call region that SAEF does not know falls back to the tracker's region.

## Vercel AI SDK

```ts
import { generateText, wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { Tracker } from "saef";
import { saefMiddleware } from "saef/ai-sdk";

const tracker = new Tracker({ region: "eu-west-1" });
const model = wrapLanguageModel({
  model: openai("gpt-4o"),
  middleware: saefMiddleware({
    region: "eu-west-1",
    tracker,
    onEstimate: (e, ctx) => console.log(ctx.modelId, e.kgCo2e, e.litres),
  }),
});

await generateText({ model, prompt: "Hello" });
console.log(tracker.totals());
```

The middleware wraps `doGenerate` and `doStream`. For streams the estimate is computed on the `finish` part. It never changes the result or throws into your call. A call that cannot be estimated, for example an embedding model, is reported to `onError` and skipped. It reads the usage of AI SDK language model specification V2, V3 and V4, including the older `promptTokens` and `completionTokens` names, and takes cache reads from `cachedInputTokens` (or `inputTokens.cacheRead`). Tested against `ai` 7.0.177.

## Regions

`region` can be a cloud region id (AWS, Azure and Google Cloud, see `regions()`), an ISO-3 country code, a US eGRID subregion, `eu-27`, or omitted for the global average. A region that matches nothing throws `SaefError` with suggestions. The default comes from the `SAEF_REGION` environment variable in `estimateFromResponse`, `Tracker` and the middleware, then falls back to the global average. If you know where your provider serves your calls, pass it, because the grid mix changes the result a lot.

Water is available for 17 regions, under two hydropower conventions (`litres` and `litresExclHydro`). Elsewhere `litres` is `null`, by design.

## What the numbers are

The factors follow the v3 lifecycle boundary. An estimate has four terms: electricity, embodied hardware and data-centre construction (charged on effective tokens) and amortised training (charged on billed tokens), in `termsKgCo2e`.

- Effective tokens are output tokens plus weighted input tokens, divided by a reference-mix normaliser.
- Never multiply a billed token count by a per-effective-token footprint yourself. Use `estimate`, or `factor(...).kgCo2ePer1mBilledTokens`.
- A model maps to class A, B or C through identifier patterns, including AWS Bedrock, Azure and Vertex AI spellings. Closed models are class proxies by pricing tier, and the result says so in `warnings`. An unmapped identifier is assigned the conservative class C and flagged. Embeddings, image, audio and video models are out of scope and throw `SaefError`.
- `kgCo2eHighCase` is a conservative bound. The development line (`developmentLineKgCo2e`) is reported separately and is never added to `kgCo2e`.
- `kwh` is the facility energy, the electricity term divided by the grid intensity.

## Estimates, not measurements

These figures are modelled estimates built from published benchmarks, grid averages and lifecycle studies. They are not measurements of any provider's infrastructure. The tables carry six significant figures as arithmetic, but the inputs support about two. Report the central value with the high case, and read the methodology before quoting them.

This package bundles the text tables of SAEF v3.4.0 (values dated 2026-10-02, identical to v3.3.0). The non-text model types of v3.4.0 are recognised but not estimated here, use the Python package for them. It covers the H100 central case. Reserved accelerator-hour factors, the B200 sensitivity and the Tier 2b seat-based estimates in the full deposit are not exposed by this API.

## EcoLogits and saef

CodeCarbon reads the power of hardware you run, so it does not fit remote APIs. For API-based LLM usage you need an estimate from token counts. EcoLogits does that per call by wrapping the provider SDKs, in Python. The `saef` package does it from the response object or from billing totals. The two draw the boundary differently (SAEF adds amortised training), so pick one and state which.

## Regenerating the data and parity tests

The factor tables in `data/saef-data.json` are generated from the Python package's CSV files with `python scripts/build-data.py <path to saef/src/saef>`. The parity tests compare this package with the Python package on 1,800 model, region and token cases. Regenerate their expected values with `PYTHONPATH=<path to saef/src> python scripts/dump-parity.py`, then run `npm test`.

## Licence

The dataset (v3.3.0 onward) is licensed under the SOMA AI Emission Factors Data Licence 1.0 (Non-Commercial), see `LICENSE`. The code is licensed under PolyForm Noncommercial 1.0.0, see `LICENSE-CODE`. Free for personal use, academic and non-profit research, journalism, charities and government. Any use by a company or freelancer, including internal ESG/CSRD reporting, needs a commercial licence: https://somaai.earth/licensing (guillermollopis@somaai.earth). Versions up to v3.2.0 stay under CC BY 4.0 (data) and MIT (code). Attribution for the data is required, see the citation below.

## Citation

Dataset (concept DOI, always the latest version): https://doi.org/10.5281/zenodo.20443585. Dataset values as in v3.2.0: https://doi.org/10.5281/zenodo.24460344

```bibtex
@misc{llopis2026somafactors,
  author    = {Llopis-Garcia, Guillermo},
  title     = {{SOMA AI Emission Factors (SAEF)}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20443585}
}
```

Method paper: Llopis-Garcia, G. (2026), Accounting for Purchased Artificial Intelligence Inference in Corporate Greenhouse Gas Inventories, a Token-Based Method. Sustainable Production and Consumption 70 (2026) 320-347. https://doi.org/10.1016/j.spc.2026.11.003

## Links

- Homepage: https://github.com/guillermollopis/soma-ai-emission-factors
- SOMA: https://somaai.earth/ai-footprint
- Paper: https://doi.org/10.1016/j.spc.2026.11.003 (preprint https://arxiv.org/abs/2606.10660)
