# SOMA AI Emission Factors (SAEF)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20443585.svg)](https://doi.org/10.5281/zenodo.20443585)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016/j.spc.2026.11.003-blue.svg)](https://doi.org/10.1016/j.spc.2026.11.003)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10660-b31b1b.svg)](https://arxiv.org/abs/2606.10660)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-lightgrey.svg)](https://somaai.earth/licensing)

SOMA AI Emission Factors (SAEF) is a versioned dataset of estimated carbon and water factors for AI inference, by model class and by region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. It covers 210 countries and economies, 27 US eGRID subregions and 143 AWS, Azure and Google Cloud regions. Every factor traces to a file and a source, and the calculator returns the file, line and column behind each number. Published by [SOMA AI](https://somaai.earth), Barcelona, under a non-commercial licence, see the Licence section. The method is published in Sustainable Production and Consumption 70 (2026) 320-347, https://doi.org/10.1016/j.spc.2026.11.003 (free preprint arXiv 2606.10660).

This repository holds the v3.5.0 tables (lifecycle boundary; text factor values dated 2026-10-02 and identical to v3.2.0 and v3.3.0, version DOI of v3.3.0 [10.5281/zenodo.24506078](https://doi.org/10.5281/zenodo.24506078); v3.4.0 adds embeddings, reranking, moderation, image generation and transcription, and v3.5.0 adds text-to-speech, realtime voice and video) and the source of the `saef` Python package. The complete deposit, with the reference calculator, the generator scripts and the supplementary documents, is on [Zenodo](https://doi.org/10.5281/zenodo.20443585).

## How can I measure the environmental impact of the AI I use?

AI that runs on hardware you control can be metered, with a power meter or with tools such as CodeCarbon that read hardware energy counters. AI you buy as a service (ChatGPT, Copilot, Claude, Gemini, API calls to a provider) cannot be metered by the customer, because providers do not publish energy for the requests you send. It can only be estimated from usage (tokens, messages or spend), an energy factor for the model size and the grid where it runs. That estimate is what SAEF is for.

| Your AI use | Fits | What you get |
|---|---|---|
| Training or inference on your own GPUs | CodeCarbon, Green Algorithms | Estimated energy and emissions of your own hardware |
| API calls made from your code | `saef` (this package, LiteLLM, LangChain, LlamaIndex and Haystack integrations) or EcoLogits | Per-call estimates |
| ChatGPT, Copilot or Claude seats across a company | SAEF factors applied to seats x messages, or the free [SOMA AI calculator](https://somaai.earth/ai-footprint/calculator) (also as a [Hugging Face Space](https://huggingface.co/spaces/GuillermoLlopis/saef-ai-footprint-estimator)) | An organisation-wide estimate for Scope 3 category 1 |
| Cloud infrastructure you rent | Your cloud provider's carbon dashboard | Emissions of your cloud account |

Full answer with a worked 250-seat Copilot example and published per-query figures: [How can I measure the environmental impact of the AI I use?](https://somaai.earth/ai-footprint/answers/how-to-measure-environmental-impact-of-ai-you-use) (Spanish: [¿Cómo puedo medir el impacto ambiental de la IA que uso?](https://somaai.earth/ai-footprint/respuestas/como-medir-impacto-ambiental-ia)). All SAEF figures are estimates, not measurements.

## Questions this dataset answers

### How can I estimate the carbon footprint of the AI tools my company uses (ChatGPT, Copilot, Claude)?

Count usage per tool, such as seats, messages or tokens, and multiply by a per-query or per-token factor for the model class and the grid region. SAEF (SOMA AI Emission Factors) publishes those factors, and the free [SOMA AI calculator](https://somaai.earth/ai-footprint/calculator) applies them to company seats. The result is an estimate with a range, not a meter reading.

### How do I calculate CO2e from LLM API token usage?

Divide billed tokens by one million and multiply by the SAEF factor F (kg CO2e per million tokens) for the model class and region: `kg CO2e = T_billed / 1e6 x F`. This one-line form assumes the dataset's default mix of input and output tokens and gives no cache credit. For a different mix, use the full form on the [formula page](https://somaai.earth/ai-footprint/how-to/llm-token-co2e-formula). For a mid-size model, F is 0.00714 kg per million tokens in EU North (Sweden) and 0.0261 kg in US East (N. Virginia).

### Which tool fits which kind of AI use?

CodeCarbon or Green Algorithms fit your own GPUs. EcoLogits or the `saef` package fit API calls made from code. The SOMA AI calculator or SAEF factors fit company seats. Your cloud provider's carbon dashboard fits infrastructure you rent.

### Can CodeCarbon track API calls to OpenAI, Anthropic or Gemini?

No. CodeCarbon reads the power of hardware you run, and its own README points users of remote APIs to EcoLogits. For API-based LLM usage you need an estimate from token counts. EcoLogits does that per call by wrapping the provider SDKs. The `saef` package does it from the response object or from billing totals (`pip install saef`, then `saef.estimate(...)` or `estimate_from_response(resp)`), with a LiteLLM callback and LangChain middleware. The two draw the boundary differently (SAEF adds amortised training), so pick one and state which.

### Where does purchased AI go in a GHG inventory?

Under Scope 3 category 1 (purchased goods and services). Label the figure as an estimate, show a low-to-high band, and name the factor version used, for example SAEF v3.5.0.

### Can the energy use of a provider's AI be measured directly?

No. Providers do not publish per-customer energy for the requests you send, so a customer can only estimate it from usage, model size and grid. SAEF is an estimation dataset built on published benchmarks, and its figures are not measurements.

### How much electricity does one AI prompt use?

SAEF estimates 0.03 to 0.37 Wh at facility level for a 1,210-token chat exchange, from a small (class A) to a large (class C) model, and 0.099 Wh for a mid-size (class B) model. Published figures for comparison are 0.24 Wh for the median Gemini Apps text prompt (Google, August 2025) and about 0.3 Wh for a GPT-4o query (Epoch AI). Long documents, coding agents and reasoning modes process far more tokens and use far more.

### How much water does ChatGPT or another AI chatbot use per prompt?

SAEF estimates 0.15 to 0.94 mL for a mid-size chat exchange, depending on the cloud region, because the region changes the water, not the watt-hours. Google reports 0.26 mL for its median Gemini Apps text prompt. The viral bottle-of-water-per-email figure (about 519 mL, Washington Post and UC Riverside, 2024) counted the water behind the electricity with mid-2024 assumptions. Details on the [water page](https://somaai.earth/ai-footprint/guides/ai-water-footprint).

### How do I measure the carbon footprint of my own AI use, as one person?

Count your messages, pick a model class and apply the grid of the region. With a mid-size model, a reasoning uplift and a US East grid, 20 chat messages a day for a year come to about 0.4 kg CO2e (band 0.2 to 1 kg), and about 0.1 kg on a Swedish grid. Coding agents, long documents and image or video generation add much more per request. A step-by-step answer is on [How can I measure the environmental impact of the AI I use?](https://somaai.earth/ai-footprint/answers/how-to-measure-environmental-impact-of-ai-you-use)

### Which AI model has the lowest footprint?

Model size class and serving region decide most of the estimate, more than the brand. Smaller models and low-carbon grids give the lowest carbon per exchange, but low-carbon grids are not always the lowest-water ones. Per-model estimates are on the [models compared page](https://somaai.earth/ai-footprint/answers/ai-models-carbon-footprint-compared).

Cite the dataset as [10.5281/zenodo.20443585](https://doi.org/10.5281/zenodo.20443585).

## Install

```bash
pip install saef
```

```python
import saef

saef.model_class_for("gpt-4o")                                   # B
r = saef.estimate(tokens=1_000_000, model="gpt-4o-mini", region="eu-west-1")
r.kg_co2e, r.kg_co2e_high_case, r.water_ml                      # estimates, with provenance in r.provenance
```

```bash
saef estimate --tokens 1000000 --model gpt-4o-mini --region eu-west-1
```

The package bundles the eleven CSV files below and has no dependencies. It is built from this repository with `python scripts/sync_data.py && python -m build`.

## Integrations

The core package has no dependencies. The integrations below estimate carbon and water for each call from the token usage your framework already reports. The region comes from the `region` argument, then from the `SAEF_REGION` environment variable, then falls back to the global average. All of them keep running totals and can append one JSON line per call with `log_path=`.

LiteLLM, installed with `pip install "saef[litellm]"`. The estimate is also attached to `response._hidden_params["saef"]`.

```python
import time
import litellm
from saef.integrations.litellm import SaefLogger

logger = SaefLogger(region="eu-west-1")
litellm.callbacks = [logger]
resp = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}], mock_response="Hi there")
time.sleep(1)  # LiteLLM runs success callbacks in a background thread
t = logger.totals()
print(t["calls"], t["input_tokens"], t["output_tokens"])
print(f"{t['kg_co2e']:.3e} kg CO2e, {t['kg_co2e_high_case']:.3e} high case, {t['water_ml']:.4f} mL")
```

```text
1 10 20
3.905e-07 kg CO2e, 6.250e-07 high case, 0.0018 mL
```

LangChain, installed with `pip install "saef[langchain]"`. This example uses a fake chat model so it runs offline. With a real model you pass the same handler in `callbacks`.

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from saef.integrations.langchain import SaefCallbackHandler

handler = SaefCallbackHandler(region="eu-west-1")
reply = AIMessage(
    content="Hi there",
    response_metadata={"model_name": "gpt-4o"},
    usage_metadata={"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500, "input_token_details": {"cache_read": 800}},
)
llm = GenericFakeChatModel(messages=iter([reply]))
llm.invoke("Hello", config={"callbacks": [handler]})
t = handler.totals()
print(t["calls"], t["input_tokens"], t["cached_input_tokens"], t["output_tokens"])
print(f"{t['kg_co2e']:.3e} kg CO2e, {t['kg_co2e_high_case']:.3e} high case")
```

```text
1 400 800 300
2.894e-05 kg CO2e, 4.433e-05 high case
```

LangChain v1 agents can use middleware instead of a callback. `SaefMiddleware` hooks `wrap_model_call` and `awrap_model_call`, records one estimate per model call, and with `attach=True` also writes the estimate to `AIMessage.response_metadata["saef"]`. It needs the `langchain` package, which the `langchain` extra installs. The example again uses a fake chat model.

```python
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from saef.integrations.langchain import SaefMiddleware

reply = AIMessage(
    content="Hi there",
    response_metadata={"model_name": "gpt-4o"},
    usage_metadata={"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500, "input_token_details": {"cache_read": 800}},
)
llm = GenericFakeChatModel(messages=iter([reply]))
mw = SaefMiddleware(region="eu-west-1", attach=True)
agent = create_agent(llm, tools=[], middleware=[mw])
out = agent.invoke({"messages": [("user", "Hello")]})
t = mw.totals()
print(t["calls"], t["input_tokens"], t["cached_input_tokens"], t["output_tokens"])
print(f"{t['kg_co2e']:.3e} kg CO2e, {t['kg_co2e_high_case']:.3e} high case")
print(out["messages"][-1].response_metadata["saef"]["kg_co2e"])
```

```text
1 400 800 300
2.894e-05 kg CO2e, 4.433e-05 high case
2.894105237117252e-05
```

LlamaIndex, installed with `pip install "saef[llamaindex]"`. `SaefEventHandler` listens to LlamaIndex's instrumentation events and records one estimate per finished chat or completion call, reading token usage from the provider's raw response. The model name comes from the raw response, then from the call's start event. Streaming calls count only if the last chunk carries usage. This example uses a fake LLM so it runs offline. With a real LLM you register the handler once and call it as usual.

```python
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_chat_callback
from saef.integrations.llamaindex import SaefEventHandler


class FakeLLM(CustomLLM):
    @property
    def metadata(self):
        return LLMMetadata(model_name="gpt-4o")

    @llm_chat_callback()
    def chat(self, messages, **kwargs):
        raw = {"model": "gpt-4o", "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 800}}}
        return ChatResponse(message=ChatMessage(role="assistant", content="Hi there"), raw=raw)

    def complete(self, prompt, formatted=False, **kwargs): ...
    def stream_complete(self, prompt, formatted=False, **kwargs): ...


handler = SaefEventHandler(region="eu-west-1").register()
FakeLLM().chat([ChatMessage(role="user", content="Hello")])
t = handler.totals()
print(t["calls"], t["input_tokens"], t["cached_input_tokens"], t["output_tokens"])
print(f"{t['kg_co2e']:.3e} kg CO2e, {t['kg_co2e_high_case']:.3e} high case")
```

```text
1 400 800 300
2.894e-05 kg CO2e, 4.433e-05 high case
```

Haystack 2.x, installed with `pip install "saef[haystack]"`. `SaefTracker` is a pass-through component. Connect `replies` from a ChatGenerator, which carries the model and usage in each reply's `meta`. For a text Generator such as `OpenAIGenerator`, also connect its `meta` output to `meta`. It outputs the replies unchanged plus `saef`, one estimate dict per reply with a model name and token usage, and keeps running totals. This example uses a fake generator so it runs offline. With a real one you add, for example, `OpenAIChatGenerator` in its place.

```python
from typing import List
from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage
from saef.integrations.haystack import SaefTracker


@component
class FakeChatGenerator:
    @component.output_types(replies=List[ChatMessage])
    def run(self, prompt: str):
        meta = {"model": "gpt-4o", "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 800}}}
        return {"replies": [ChatMessage.from_assistant("Hi there", meta=meta)]}


pipe = Pipeline()
pipe.add_component("llm", FakeChatGenerator())
pipe.add_component("saef", SaefTracker(region="eu-west-1"))
pipe.connect("llm.replies", "saef.replies")
result = pipe.run({"llm": {"prompt": "Hello"}})
print(result["saef"]["saef"][0]["kg_co2e"])
t = pipe.get_component("saef").totals()
print(t["calls"], t["input_tokens"], t["cached_input_tokens"], t["output_tokens"])
print(f"{t['kg_co2e']:.3e} kg CO2e, {t['kg_co2e_high_case']:.3e} high case")
```

```text
2.894105237117252e-05
1 400 800 300
2.894e-05 kg CO2e, 4.433e-05 high case
```

All figures are estimates from token counts, not measurements.

Response objects, with no extra install and no SDK import. `estimate_from_response` reads OpenAI Chat Completions, OpenAI Responses and Anthropic Messages objects, or the same shapes as dicts. Cache reads get the cache credit and cache writes count as ordinary input.

```python
from types import SimpleNamespace
from saef.integrations.openai import estimate_from_response

resp = SimpleNamespace(
    type="message", model="claude-sonnet-4-5",
    usage=SimpleNamespace(input_tokens=200, output_tokens=150, cache_read_input_tokens=1800, cache_creation_input_tokens=0),
)
e = estimate_from_response(resp, region="eu-west-1")
print(e.model_class, f"{e.kg_co2e:.3e}", f"{e.kg_co2e_high_case:.3e}", f"{e.water_ml:.4f}")
```

```text
B 2.231e-05 3.235e-05 0.0785
```

Every figure is an estimate. Models the dataset treats as out of scope, such as music generation, and calls whose usage does not fit the model type, are skipped and counted in `totals()["skipped_calls"]`.

Example notebooks: [`examples/quickstart.ipynb`](examples/quickstart.ipynb), and a runnable copy on Kaggle, [Estimate the carbon footprint of your LLM API calls](https://www.kaggle.com/code/guillermollopis/estimate-the-carbon-footprint-of-your-llm-api-call).

## Cost and carbon side by side

Cost and carbon side by side. If your team optimises AI cost as well as carbon, saef can put both on the same record. saef does not price tokens and has no price table. It takes the cost the framework already reports. With LiteLLM that is `response_cost`, in USD, and the logger stores it as `cost` and `cost_currency` next to the estimate. Calls where LiteLLM reports no cost simply have no cost fields.

```python
import time
import litellm
from saef.integrations.litellm import SaefLogger

logger = SaefLogger(region="eu-west-1")
litellm.callbacks = [logger]
resp = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}], mock_response="Hi there")
time.sleep(1)
rec = resp._hidden_params["saef"]
print(rec["kg_co2e"], rec["cost"], rec["cost_currency"])
t = logger.totals()
print(f"{t['cost']:.6f} {t['cost_currency']}, {t['g_co2e_per_usd']:.4f} g CO2e per USD")
```

`g_co2e_per_usd` is the central estimate in grams divided by the cost, over the calls that have a cost, and is `None` when the total cost is zero. To total records you logged or stored yourself, use `saef.summarize(records)`, which accepts the per-call dicts or `Estimate` objects and returns the same cost fields.

## MCP server

SAEF ships an MCP server so an AI agent can estimate the carbon, energy and water of AI usage on request. It runs locally over stdio and needs the optional `mcp` extra. The core package stays dependency-free.

```bash
pip install "saef[mcp]"
```

Claude Code.

```bash
claude mcp add saef -- saef-mcp
```

Claude Desktop, in `claude_desktop_config.json`.

```json
{
  "mcpServers": {
    "saef": { "command": "saef-mcp" }
  }
}
```

Cursor, in `.cursor/mcp.json`.

```json
{
  "mcpServers": {
    "saef": { "command": "saef-mcp" }
  }
}
```

If `saef-mcp` is not on your PATH, use `uvx --from "saef[mcp]" saef-mcp` as the command instead.

Tools.

- `estimate_footprint(tokens, model, region="global")` returns carbon in kg and g CO2e with a central and a high case, facility energy in Wh, water in mL where the region has a water factor, the model class, the region used, the dataset version and a provenance trail.
- `estimate_chat_usage(messages, model, region="global")` returns the same for chat exchanges, using the dataset's short-chat default of 1,210 tokens per exchange (928 input, 282 output).
- `estimate_modality(unit_count, unit, model, region="global", megapixels=None)` returns the same, with low and high, for embeddings, reranking, moderation, images, transcription, speech, voice and video.
- `model_class(model)` returns the class letter (A small, B mid-size, C large) and a description.
- `list_regions()` returns the accepted region codes, with names for cloud regions and countries.
- `compare_regions(tokens, model, regions)` returns carbon and water per region, sorted from lowest to highest carbon.

The server also exposes the resource `saef://about` with the licence, DOI and method paper. Every figure is an estimate, not a measurement.

Example prompt, "Using SAEF, what is the carbon and water footprint of 2 million gpt-4o tokens a month in eu-north-1 compared with us-east-1?"

## Files

| File | Rows | What it contains |
|---|---|---|
| `cloud_region_factors.csv` | 143 | Class A, B and C factors for each AWS, Azure and Google Cloud region, resolved to a grid, with the confidence of the region mapping |
| `grid_intensity_all.csv` | 240 | Grid carbon intensity (kg CO2e per kWh) for 210 countries and economies, 27 US eGRID subregions, EU-27, the world average and one subnational row, with the class factors at each grid |
| `factors_v3_lifecycle.csv` | 102 | Class factors by class, GPU (H100 central, B200 sensitivity) and printed region, with the energy and the four terms |
| `model_factors.csv` | 32 | Factors for the 18 measured open-weight models on H100 and B200, from the ML.ENERGY Leaderboard v3 |
| `model_class_mapping.csv` | 287 | Identifier patterns, including AWS Bedrock, Azure and Vertex AI spellings, mapped to a model type (`model_type`, default `text`) and its class, plus the class C default for unmapped names |
| `modality_factors.csv` | 101 | GPU, server and facility energy, lifecycle terms, and CO2e and water at reference regions per functional unit of the non-text model types, with low, central and high |
| `modality_sources.csv` | 35 | Sources of `modality_factors.csv`, with licence and role |
| `region_water_factors.csv` | 17 | On-site and off-site water factors under both hydropower conventions, with AWARE 2.0 scarcity weighting |
| `region_grid_crosswalk.csv` | 17 | Printed region ids mapped to grids |
| `token_basis_constants.csv` | 7 | Weights that convert billed tokens to effective tokens |
| `reserved_capacity_factors.csv` | 3 | Factors per accelerator-hour for provisioned capacity |
| `src/saef/` | n/a | The Python package: a standard-library calculator over these files |

Three model classes are used: A (small, about 8B to 30B), B (standard, about 70B) and C (frontier). Every closed commercial model is class-only, assigned by the provider's pricing tier, and must be disclosed as an assumption.

## Method

An estimate has four terms. Electricity, embodied hardware and data-centre construction are charged on effective tokens. Amortised training is charged on billed tokens.

```
kg CO2e = T_eff / 1e6 x inference footprint  +  T_billed / 1e6 x training term
T_eff   = (T_out + 0.30 x T_in_uncached + 0.05 x T_in_cached) / 1.339
```

Never multiply a billed token count by an inference footprint. At the default mix that overstates the result by about 3.1 times. Where only a billed total is known, use the default-mix factor per billed token, which already carries the conversion (0.302465 effective tokens per billed token at an 85% input share with no cache credit).

- An unmapped identifier is assigned class C and flagged. Embeddings, reranking, moderation, images, transcription, speech, voice and video have their own model types, see below. Audio chat, music, OCR and fine-tuning jobs are out of scope.
- Non-text and out-of-scope rows win over any text-model row that also matches, so a name like `gpt-4o-mini-transcribe` never receives a text class. Bedrock camelCase names such as `TitanEmbeddingV2-Text` are split at case boundaries before matching.
- Token basis. Cache-write tokens (Anthropic `cache_creation_input_tokens`, Bedrock `cacheWriteInputTokens`) count as uncached input, weight W_in. Reasoning and thinking tokens count as output. OpenAI, Anthropic and Bedrock already include them in billed output, so add nothing. Gemini reports `thoughtsTokenCount` separately from `candidatesTokenCount`, so add it to output.
- Unknown or global regions use the global-average grid. Water has no fallback, and covers 17 regions only.
- Each result carries a conservative high case. A development line is reported separately and never added to a total.
- Six significant figures are arithmetic, not evidence. The inputs support about two. Report the central value with the high case.

The numbers are estimates, not measurements.

## Model types other than text (since dataset v3.4.0, speech, voice and video since v3.5.0)

Text models are charged on tokens by class A, B or C. Ten more model types have their own unit and class, in `modality_factors.csv`. Every value there is an estimate, labelled `derived` (arithmetic on published benchmark results and documented facts) or `assumption` (an author judgement with a stated range). No provider publishes energy for these services.

| Model type | Class | Unit | Facility Wh, central (low to high) | g CO2e, eu-west-1 | g CO2e, us-east-1 |
|---|---|---|---|---|---|
| Text embeddings | E-small | 1,000 input tokens | 0.00057 (9e-05 to 0.052) | 0.00017 (2.7e-05 to 0.016) | 0.00018 (2.9e-05 to 0.017) |
|  | E-large | 1,000 input tokens | 0.016 (0.0019 to 0.45) | 0.0049 (0.00057 to 0.14) | 0.0052 (0.00061 to 0.15) |
| Image embeddings | E-small | image | 0.00015 (2.3e-05 to 0.0023) | 4.4e-05 (7e-06 to 0.00069) | 4.7e-05 (7.5e-06 to 0.00074) |
|  | E-large | image | 0.016 (0.00048 to 0.45) | 0.0049 (0.00015 to 0.14) | 0.0052 (0.00016 to 0.15) |
| Reranking | R-small | search unit | 0.029 (0.00075 to 0.58) | 0.0087 (0.00023 to 0.17) | 0.0093 (0.00024 to 0.19) |
|  | R-large | search unit | 0.48 (0.0094 to 23) | 0.15 (0.0028 to 6.8) | 0.16 (0.003 to 7.3) |
| Moderation | M-A | call | 0.0081 (0.0002 to 0.059) | 0.0024 (6.1e-05 to 0.018) | 0.0026 (6.5e-05 to 0.019) |
|  | M-B | call | 0.026 (0.00034 to 0.16) | 0.008 (0.0001 to 0.049) | 0.0085 (0.00011 to 0.053) |
| Token-billed images | IMG-TOK-B | 1024x1024 image, high quality | 0.73 (0.45 to 2.7) | 0.22 (0.14 to 0.82) | 0.24 (0.15 to 0.88) |
|  | IMG-TOK-A | image, size and quality unknown | 0.54 (0.23 to 2.7) | 0.16 (0.069 to 0.82) | 0.17 (0.074 to 0.88) |
| Per-image diffusion | D-small | image up to 1.05 MP | 0.54 (0.23 to 0.87) | 0.16 (0.069 to 0.26) | 0.17 (0.074 to 0.28) |
|  | D-large | image up to 1.05 MP | 3.2 (2.4 to 4.4) | 0.97 (0.73 to 1.3) | 1 (0.78 to 1.4) |
| Transcription | STT-ENC | audio minute | 0.06 (0.01 to 0.34) | 0.018 (0.0031 to 0.1) | 0.019 (0.0033 to 0.11) |
|  | STT-LLM-A | audio minute | 0.027 (0.012 to 0.054) | 0.0081 (0.0037 to 0.016) | 0.0087 (0.0039 to 0.017) |
|  | STT-LLM-B | audio minute | 0.088 (0.041 to 0.15) | 0.027 (0.012 to 0.045) | 0.028 (0.013 to 0.048) |
| Text-to-speech | TTS-B | 1,000 characters | 0.083 (0.039 to 0.54) | 0.025 (0.012 to 0.16) | 0.027 (0.013 to 0.18) |
|  | TTS-B | audio minute | 0.075 (0.041 to 0.41) | 0.022 (0.012 to 0.12) | 0.024 (0.013 to 0.13) |
| Realtime voice, turn-based | RT-A | 1,000 audio output tokens | 0.059 (0.032 to 0.12) | 0.018 (0.0097 to 0.036) | 0.019 (0.01 to 0.038) |
|  | RT-B | 1,000 audio output tokens | 0.19 (0.11 to 0.33) | 0.058 (0.033 to 0.098) | 0.062 (0.035 to 0.11) |
|  | RT-A | conversation minute (fallback) | 0.11 (0.02 to 1.3) | 0.033 (0.0059 to 0.38) | 0.035 (0.0063 to 0.41) |
|  | RT-B | conversation minute (fallback) | 0.36 (0.066 to 3.5) | 0.11 (0.02 to 1.1) | 0.11 (0.021 to 1.1) |
|  | RT-C | conversation minute (fallback) | 1.3 (0.13 to 16) | 0.4 (0.04 to 4.9) | 0.43 (0.043 to 5.3) |
| Realtime voice, full-duplex | FD-A | active session minute | 0.084 (0.026 to 0.19) | 0.025 (0.008 to 0.057) | 0.027 (0.0085 to 0.061) |
| Video | V-A | second at 720p | 24 (4.8 to 77) | 7.1 (1.4 to 23) | 7.6 (1.5 to 25) |
|  | V-B | second at 720p | 83 (22 to 338) | 25 (6.5 to 102) | 27 (7 to 109) |
|  | V-B | second at 1080p | 186 (26 to 1452) | 56 (7.8 to 437) | 60 (8.4 to 468) |
|  | V-C | second at 720p | 206 (83 to 844) | 62 (25 to 254) | 67 (27 to 272) |

How to map provider usage to a unit.

- **Text embeddings** turn text into vectors. Use OpenAI `usage.prompt_tokens`, Bedrock Titan `inputTokenCount`, Cohere `meta.billed_units.input_tokens` or the Google token count, as `input_tokens`. Class E-small covers list prices of USD 0.05 per million tokens or less and disclosed sizes up to 1B. Everything else is E-large.
- **Image embeddings** use `images`. Where the provider bills image tokens (Cohere Embed 4), pass them as `input_tokens`.
- **Reranking** scores documents against a query. Pass the tokens scored, summed over all query and document pairs, as `input_tokens`, or billed `search_units` when tokens are unknown (30,000 tokens per unit is assumed, 5,000 to 50,000).
- **Moderation** uses `input_tokens`, or `calls` when text length is unknown (500 tokens per call is assumed, 21 to 2,500). OpenAI moderation is free and absent from billing exports, so count calls in your own logs.
- **Token-billed image generation** (gpt-image, Gemini image). Pass `usage.output_tokens` or `candidatesTokenCount` as `image_output_tokens`, input text and image tokens as `input_tokens`, and the number of `images`. An image token is charged as one output token of the model's text class, and a closed model is never charged below an open diffusion model: the central value is the higher of token parity and the D-small value (0.45 Wh per 1024x1024 image at server level), and the low value never falls below the smallest model ML.ENERGY measured. The high value reaches class C. Without an image count, the floor assumes 1,056 tokens per image and at least one image. Count-only units are `images`, `images_1024_low`, `images_1024_medium`, `images_1024_high` and `images_4k`.
- **Per-image diffusion** (DALL-E, Imagen, Titan Image, Nova Canvas, Stable Diffusion, FLUX) uses `images`, with `megapixels=` above 1024x1024 (linear scaling, an assumption). Closed models with no disclosed size are D-large.
- **Transcription.** Whisper and managed services use `audio_minutes` or `audio_seconds`. For gpt-4o-transcribe models pass the reported audio `input_tokens` and text `output_tokens`. Minutes are a fallback that assumes 1,000 audio tokens per minute.
- **Text-to-speech.** Pass reported `audio_output_tokens` (gpt-4o-mini-tts, Gemini TTS) with text `input_tokens`, else billed `characters`, else `audio_minutes` (900 characters per minute is assumed, English). ElevenLabs credits convert to characters first. There is one class, TTS-B: no open-licence source supports a lower class for compact voices.
- **Realtime voice, turn-based** (gpt-realtime, Gemini Live, Nova Sonic, ChatGPT Advanced Voice). Use tokens: `input_tokens` (audio and text), `cached_input_tokens`, `text_output_tokens` and `audio_output_tokens`. The conversation is re-sent on every response, and only the token counts show it. Without tokens, `conversation_minutes` is an order-of-magnitude fallback, and `user_audio_minutes`, `assistant_audio_minutes` and `context_minutes` are its parts. Per-minute rows use OpenAI token rates.
- **Realtime voice, full-duplex** (gpt-live, advanced-voice-bidi). Use `active_session_minutes`. Standby minutes are zero. Backend model tokens (for example a codex model) are billed separately: estimate them under the backend model name.
- **Video.** Use `video_seconds` (720p) or `video_seconds_480p`, `video_seconds_1080p`, `video_seconds_2160p`, or `video_clips` (8 seconds at 720p is assumed). Classes follow the price per second at 720p: V-A up to USD 0.12, V-B to 0.30, V-C above and for unmapped names. The serving factor of 0.5 (0.2 to 1.0) is an assumption that moves every video value, and so are the resolution multipliers (480p 0.4, 1080p 2.25, 2160p 3.0). `gemini-omni-flash` billed in video seconds is video, billed in tokens it is text class B.

```python
import saef

saef.model_type_for("text-embedding-3-small")                                 # ('embedding_text', 'E-small')
e = saef.estimate_modality("whisper-1", 60, "audio_minutes", region="eu-west-1")
e.energy_wh, e.kg_co2e_low, e.kg_co2e, e.kg_co2e_high, e.water_ml             # facility Wh, kg CO2e, mL
saef.estimate_modality("veo-3.1-fast", 8, "video_seconds_1080p", region="us-east-1")
saef.estimate(model="gpt-image-1.5", region="eu-west-1",
              usage={"input_tokens": 50, "image_output_tokens": 4160, "images": 1})
```

```bash
saef modality --model dall-e-3 --unit images --count 100 --region eu-west-1
```

- **Routing.** A billing unit that names a model type (images, audio minutes, search units, video seconds) decides before the model name. If the name does not fit the unit, the conservative default class of the unit is used and flagged in `warnings`.
- **Energy chain.** Values are stored at GPU level and carried through the text chain unchanged: x 1.5 host x 1.1 idle = server, x 1.2 PUE = facility, then grid and water per region.
- **Lifecycle.** Construction and embodied hardware are charged per server kWh at the class A ratio. **Training is excluded** for every model type in this section, because there is no basis for it. Disclose it as excluded.
- **Ranges.** `low` and `high` are the class range of the energy at central overhead. `kg_co2e_high_case` is the central energy at the conservative overhead, as for text. The two are not combined. Ranges span two orders of magnitude for embeddings and Whisper, so report them with the central value.
- **Excluded.** Audio chat completion models (gpt-audio), music, OCR and fine-tuning have no factor and raise an error. Long video clips (20 seconds and more) are understated, because energy grows faster than length. Free or failed generations that no export lists are not counted.

## Citation

```bibtex
@misc{llopis2026somafactors,
  author       = {Llopis-Garcia, Guillermo},
  title        = {{SOMA AI Emission Factors (SAEF)}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3.5.0},
  doi          = {10.5281/zenodo.20443585},
  url          = {https://doi.org/10.5281/zenodo.20443585}
}

@article{llopis2026accounting,
  author        = {Llopis-Garcia, Guillermo},
  title         = {Accounting for purchased artificial intelligence inference in corporate greenhouse gas inventories: A token-based method},
  journal       = {Sustainable Production and Consumption},
  volume        = {70},
  pages         = {320--347},
  year          = {2026},
  publisher     = {Elsevier},
  doi           = {10.1016/j.spc.2026.11.003},
  url           = {https://doi.org/10.1016/j.spc.2026.11.003},
  eprint        = {2606.10660},
  archivePrefix = {arXiv}
}
```

Pin a dataset in an inventory by the version DOI. The concept DOI always resolves to the latest version.

## Links

- Kaggle notebook: https://www.kaggle.com/code/guillermollopis/estimate-the-carbon-footprint-of-your-llm-api-call
- Hugging Face Space, try the estimator in the browser: https://huggingface.co/spaces/GuillermoLlopis/saef-ai-footprint-estimator

- Zenodo (concept DOI, always latest version): https://doi.org/10.5281/zenodo.20443585
- Zenodo v3.2.0 record: https://zenodo.org/records/24460344
- Zenodo v3.0.0 record: https://zenodo.org/records/23371447
- Website: https://somaai.earth/ai-footprint/factors
- Paper (Sustainable Production and Consumption, 2026): https://doi.org/10.1016/j.spc.2026.11.003
- Paper preprint (free, arXiv): https://arxiv.org/abs/2606.10660
- Methodology: https://somaai.earth/ai-footprint/methodology
- Developer docs / API: https://somaai.earth/ai-footprint/data
- Free AI carbon footprint calculator for organisations: https://somaai.earth/ai-footprint/calculator
- How to calculate a company's AI carbon footprint (Scope 3 method): https://somaai.earth/ai-footprint/guides/ai-usage-csrd-scope-3
- Why published AI footprint figures differ: https://somaai.earth/ai-footprint/guides/why-ai-footprint-estimates-differ
- AI carbon footprint tools compared: https://somaai.earth/ai-footprint/guides/ai-carbon-footprint-tools-compared
- AI environmental impact statistics, with sources: https://somaai.earth/ai-footprint/answers/ai-environmental-impact-statistics
- Guía en español: https://somaai.earth/ai-footprint/guias/huella-de-carbono-ia-empresa
- How-to: calculate your company's AI carbon footprint, step by step: https://somaai.earth/ai-footprint/how-to/calculate-company-ai-carbon-footprint
- How-to: formula to estimate CO2e from LLM token usage: https://somaai.earth/ai-footprint/how-to/llm-token-co2e-formula
- How-to: track emissions from AI API calls (Python and JS examples using this dataset): https://somaai.earth/ai-footprint/how-to/track-ai-api-emissions
- AI carbon footprint by country and cloud region (209 countries ranked): https://somaai.earth/ai-footprint/answers/ai-carbon-footprint-by-country
- AI water footprint by cloud region: https://somaai.earth/ai-footprint/answers/ai-water-footprint-by-region
- Policy note, why AI providers should disclose where prompts run: https://somaai.earth/ai-footprint/answers/why-ai-providers-should-disclose-where-prompts-run
- Contact: guillermollopis@somaai.earth

## License

The data (v3.3.0 onward) is licensed under the SOMA AI Emission Factors Data Licence 1.0 (Non-Commercial), see [LICENSE](LICENSE). The code in `src/` and `js/` is licensed under PolyForm Noncommercial 1.0.0, see [LICENSE-CODE](LICENSE-CODE). Free for personal use, academic and non-profit research, journalism, charities and government. Any use by a company or freelancer, including internal ESG/CSRD reporting, needs a commercial licence: https://somaai.earth/licensing (guillermollopis@somaai.earth). Versions up to v3.2.0 stay under CC BY 4.0 (data) and MIT (code). Attribution: Llopis-Garcia, G. (2026), SOMA AI Emission Factors (SAEF), https://doi.org/10.5281/zenodo.20443585.
