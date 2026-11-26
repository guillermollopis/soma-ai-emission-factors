# Changelog

## 3.8.0 (dataset SAEF v3.4.0, release date to be set)

- Titan embedding prices and Cohere rerank prices (including Rerank 5, priced per 1M tokens) were rechecked on the providers' pages on 2026-11-26. No class changed.
- New model types. `modality_factors.csv` (with `modality_sources.csv`) adds text and image embeddings, reranking, moderation, token-billed and per-image image generation, and transcription, per functional unit with low, central and high. Values are stored at GPU level and use the text chain unchanged. Construction and embodied are charged per server kWh at the class A ratio. Training is excluded for these types.
- `model_class_mapping.csv` has a new column `model_type` (default `text`), and rows that were out of scope now point to a model type and class where one exists (254 rows). Text-to-speech, realtime voice and video stay out of scope. No text row and no text factor changed.
- New `saef.estimate_modality(model, unit_count, unit, region)`, `saef.model_type_for(model)`, `ModalityEstimate`, CLI `saef modality` and MCP tool `estimate_modality`. `saef.estimate(usage={...})` routes a usage record by billing unit before name, and `saef.estimate(tokens=...)` on an embedding model now returns an estimate instead of an error. `estimate_from_response` reads image responses (with the image count) and transcription durations.
- The JavaScript package gets `estimateModality` and `estimateModalityUsage` with the same factors, routing, warnings and errors as Python (`js/src/modality.ts`, held to the Python output by `js/test/modality-parity.test.ts`).
- Behaviour change: the integrations now count embedding calls instead of skipping them.
- Closed image models are never charged below open diffusion models. An image output token counts 1/D effective tokens, as a text output token does.
- Out-of-scope rows now win whenever they match, instead of competing on literal prefix length. `gpt-4o-mini-transcribe-*` used to get Class A through `gpt-4o-mini`. Same fix for `gpt-4o-mini-audio-*`, `gpt-4o-mini-realtime-*`, `gpt-4o-mini-image`, `gemini-*-flash-live`, `gemini-*-flash-*-tts` and `gemini-*-transcribe`.
- Identifiers are also tried with camelCase split into hyphens, so Bedrock names like `TitanEmbeddingV2-Text` and `TitanEmbeddingsG1-Image` match the Titan embedding and image rows instead of the class C default.
- New out-of-scope patterns: `advanced-voice` (ChatGPT voice mode), `gpt-*-tts`, `gpt-*-audio`, `gpt-*-realtime`, `gpt-*-transcribe`, `gpt-*-image`, `gemini-*-native-audio`. No factor changed.
- Documented the token basis for cache-write tokens (uncached input) and reasoning tokens (output, with Gemini `thoughtsTokenCount` added separately).

## 3.7.0 (2026-11-24)

- Licence change. Package code is now PolyForm Noncommercial 1.0.0 and the bundled dataset v3.3.0 is under the SOMA AI Emission Factors Data Licence 1.0 (Non-Commercial). Free for personal use, academic and non-profit research, journalism, charities and government. Any use by a company or freelancer, including internal ESG/CSRD reporting, needs a commercial licence: https://somaai.earth/licensing (guillermollopis@somaai.earth).
- Dataset v3.3.0 has the same values as v3.2.0, only the version label and licence changed.
- Versions up to 3.6.0 (dataset v3.2.0 and earlier) stay under CC BY 4.0 (data) and MIT (code).

## 3.6.0 (2026-11-23)

- Bundles SAEF dataset v3.2.0 (version DOI 10.5281/zenodo.24460344). The model class mapping grows from 187 to 220 patterns. The 33 new patterns cover 17 more providers, among them IBM Granite, ByteDance Seed, StepFun, Upstage Solar, Sarvam, Liquid AI, Inception Mercury and Sakana, so these ids now resolve to a named row instead of the class C default. No existing row and no factor changed.

## 3.5.0 (2026-11-21)

- Cost beside carbon. When the framework already reports the price of a call, saef records it next to the estimate and never prices tokens itself. `saef.integrations.litellm.SaefLogger` reads LiteLLM's `response_cost` (from the callback kwargs, then from the standard logging object) and adds `cost` and `cost_currency` ("USD") to the per-call record, the `response._hidden_params["saef"]` dict and the `log_path` lines. `totals()` adds `cost`, `cost_currency`, `calls_with_cost` and `g_co2e_per_usd` (grams CO2e per USD, over the calls that have a cost; None when total cost is zero). Calls without a reported cost are unchanged.
- `Tracker.record()` takes optional `cost` and `cost_currency`.
- Added `saef.summarize(records)`, totals over Estimate objects or the per-call dicts, with the same cost fields.
- Dataset unchanged, still v3.1.0.

## 3.4.0 (2026-11-20)

- `saef[mcp]` now pins `mcp<1.30`: mcp 1.30 with the separate `mcp-types` package fails on `import mcp`, which broke fresh installs of the MCP server.

- Added `saef.integrations.llamaindex.SaefEventHandler`, a LlamaIndex instrumentation event handler. It records one estimate per finished chat or completion call from the provider's raw response, exposes `totals()` and `reset()`, and `register()` adds it to the LlamaIndex root dispatcher. New optional extra `llamaindex`.
- Added `saef.integrations.haystack.SaefTracker`, a Haystack 2.x pass-through component to place after a Generator or ChatGenerator. It reads `model` and `usage` from reply metadata, outputs the replies and a `saef` list of estimates, and keeps running totals. New optional extra `haystack`.
- Dataset unchanged, still v3.1.0.

## 3.3.2 (2026-11-18)

- The method paper is published: Sustainable Production and Consumption 70 (2026) 320-347, doi 10.1016/j.spc.2026.11.003. `saef.citation()` now returns an `@article` entry with the DOI, and the MCP server cites the journal version.
- PyPI description and keywords now say what the package is used for, tracking the footprint of OpenAI, Anthropic, LiteLLM and LangChain calls. README opens with the response-object example.
- Dataset unchanged, still v3.1.0.

## 3.3.1 (2026-11-17)

- Pin the `mcp` extra to `mcp>=1.2,<2`. mcp 2.x renamed FastMCP, so `saef-mcp` failed to start on a fresh install of 3.3.0.

## 3.3.0

- Added an MCP server, `saef-mcp` (module `saef.mcp_server`), so AI agents in Claude Desktop, Claude Code, Cursor and other MCP clients can estimate carbon, energy and water. Tools are `estimate_footprint`, `estimate_chat_usage`, `model_class`, `list_regions` and `compare_regions`, plus the resource `saef://about`.
- New optional extra `mcp` (`pip install "saef[mcp]"`). The core package is still dependency-free and imports `mcp` only inside `saef.mcp_server`.
- Dataset unchanged, still v3.1.0.

## 3.2.0

- Added `saef.integrations.langchain.SaefMiddleware` for LangChain v1 agents. It hooks `wrap_model_call` and `awrap_model_call`, records one estimate per model call, exposes `totals()` and `reset()`, and with `attach=True` writes the estimate to `AIMessage.response_metadata["saef"]`. The existing `SaefCallbackHandler` is unchanged.
- The `langchain` extra now installs `langchain>=1.0` (which includes `langchain-core`).
- Added `examples/quickstart.ipynb` and `examples/kaggle_notebook.ipynb`, executed notebooks that estimate a call, compare regions, use the LiteLLM callback and show how to cite.
- Dataset unchanged, still v3.1.0.

## 3.1.0

- Added LiteLLM, LangChain callback and response-object integrations.
- Dataset v3.1.0, a corrective release of v3.0.0 (Quebec regions mapped to the Quebec grid, zero-grid fallback for the Central African Republic).
