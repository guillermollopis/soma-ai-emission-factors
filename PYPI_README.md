# saef

Python access to SAEF, the SOMA AI Emission Factors. SAEF is a versioned dataset of estimated carbon and water factors for AI inference, by model class and region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. This package bundles the CSV files and gives you a small API and command line on top of them. It has no dependencies beyond the standard library.

Use it to track the estimated carbon and water footprint of OpenAI, Anthropic and other LLM API calls from the token usage each response already reports, with a LiteLLM callback, a LangChain handler, an MCP server, or one function call on a response object. The method is peer reviewed, published in Sustainable Production and Consumption 70 (2026) 320-347 (https://doi.org/10.1016/j.spc.2026.11.003).

Every number comes with the file, column and row it was read from, so you can trace an estimate back to the dataset.

## Install

```bash
pip install saef
```

For pandas users, `pip install "saef[pandas]"` adds the optional extra. The framework integrations below use `saef[litellm]`, `saef[langchain]`, `saef[llamaindex]` and `saef[haystack]`.

## Track OpenAI and Anthropic API calls

```python
from saef.integrations.openai import estimate_from_response

resp = client.chat.completions.create(model="gpt-4o-mini", messages=[...])  # or client.messages.create(...) with Anthropic
e = estimate_from_response(resp, region="eu-west-1")
print(e.kg_co2e, e.kg_co2e_high_case, e.water_ml)  # kg CO2e central and high case, water in mL
```

No SDK import is needed, so the same call works on OpenAI Chat Completions, OpenAI Responses and Anthropic Messages objects, or the same shapes as dicts. Running totals per model and a JSON lines log are covered under Integrations below.

## Example

```python
import saef

print(saef.model_class_for("gpt-4o"))             # B
print(saef.factor("B", "eu-west-1")["kg_co2e_per_1m_billed_tokens"])

r = saef.estimate(tokens=1_000_000, model="gpt-4o-mini", region="eu-west-1")
print(r.kg_co2e, r.kg_co2e_high_case)             # central and conservative high case, kg CO2e
print(r.water_ml)                                 # mL, or None where the region has no water factor
for s in r.provenance[:3]:
    print(s["file"], s["line"], s["column"])      # the file, line and column behind each number
print(len(saef.regions()), saef.__version__, saef.DATASET_VERSION)  # 143 cloud regions, 3.8.0, 3.4.0
print(saef.citation())
```

Command line:

```bash
saef estimate --tokens 1000000 --model gpt-4o-mini --region eu-west-1
saef estimate --output-tokens 2e6 --input-tokens 10e6 --model gpt-4o --provider azure --region swedencentral
saef factor --model-class B --region us-east-1
saef modality --model whisper-1 --unit audio_minutes --count 60 --region eu-west-1
saef regions --provider aws
saef cite
```

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

Every figure is an estimate. Models the dataset treats as out of scope, such as text-to-speech, and calls whose usage does not fit the model type, are skipped and counted in `totals()["skipped_calls"]`.

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
- `estimate_modality(unit_count, unit, model, region="global", megapixels=None)` returns the same, with low and high, for embeddings, reranking, moderation, images and transcription.
- `model_class(model)` returns the class letter (A small, B mid-size, C large) and a description.
- `list_regions()` returns the accepted region codes, with names for cloud regions and countries.
- `compare_regions(tokens, model, regions)` returns carbon and water per region, sorted from lowest to highest carbon.

The server also exposes the resource `saef://about` with the licence, DOI and method paper. Every figure is an estimate, not a measurement.

Example prompt, "Using SAEF, what is the carbon and water footprint of 2 million gpt-4o tokens a month in eu-north-1 compared with us-east-1?"

## What the numbers are

The factors follow the v3 lifecycle boundary. An estimate has four terms: electricity, embodied hardware and data-centre construction (charged on effective tokens) and amortised training (charged on billed tokens).

- Effective tokens are output tokens plus weighted input tokens, divided by a reference-mix normaliser. Pass `tokens` for a billed total, which applies the default input share of 85% and no cache credit. Pass `output_tokens`, `input_tokens` and `cached_input_tokens` when your billing export gives the split.
- Never multiply a billed token count by a per-effective-token footprint yourself. Use `estimate`, or the `kg_co2e_per_1m_billed_tokens` value from `factor`.
- A model maps to class A, B or C through the text rows of 254 identifier patterns, including AWS Bedrock, Azure and Vertex AI spellings. Closed models are class proxies by pricing tier, and the result says so in `warnings`. An unmapped identifier is assigned the conservative class C and flagged. Embeddings, reranking, moderation, images and transcription have their own model types, see below. Text-to-speech, realtime voice and video models are out of scope and raise an error.
- `region` can be a cloud region id (143 AWS, Azure and Google Cloud regions), an ISO-3 country code (210 countries and economies), a US eGRID subregion, `eu-27`, or `None` for the global average. A region that matches nothing raises `saef.SaefError` with suggestions.
- Water covers 17 regions, under two hydropower conventions (`water_ml` and `water_ml_excl_hydro`). Elsewhere `water_ml` is `None`, by design.
- `kg_co2e_high_case` is a conservative bound. The development line is reported separately and is never added to `kg_co2e`.

## Model types other than text (since dataset v3.4.0)

Text models are charged on tokens by class A, B or C. Six more model types have their own unit and class, in `modality_factors.csv`. Every value there is an estimate, labelled `derived` (arithmetic on published benchmark results and documented facts) or `assumption` (an author judgement with a stated range). No provider publishes energy for these services.

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

How to map provider usage to a unit.

- **Text embeddings** turn text into vectors. Use OpenAI `usage.prompt_tokens`, Bedrock Titan `inputTokenCount`, Cohere `meta.billed_units.input_tokens` or the Google token count, as `input_tokens`. Class E-small covers list prices of USD 0.05 per million tokens or less and disclosed sizes up to 1B. Everything else is E-large.
- **Image embeddings** use `images`. Where the provider bills image tokens (Cohere Embed 4), pass them as `input_tokens`.
- **Reranking** scores documents against a query. Pass the tokens scored, summed over all query and document pairs, as `input_tokens`, or billed `search_units` when tokens are unknown (30,000 tokens per unit is assumed, 5,000 to 50,000).
- **Moderation** uses `input_tokens`, or `calls` when text length is unknown (500 tokens per call is assumed, 21 to 2,500). OpenAI moderation is free and absent from billing exports, so count calls in your own logs.
- **Token-billed image generation** (gpt-image, Gemini image). Pass `usage.output_tokens` or `candidatesTokenCount` as `image_output_tokens`, input text and image tokens as `input_tokens`, and the number of `images`. An image token is charged as one output token of the model's text class, and a closed model is never charged below an open diffusion model: the central value is the higher of token parity and the D-small value (0.45 Wh per 1024x1024 image at server level), and the low value never falls below the smallest model ML.ENERGY measured. The high value reaches class C. Without an image count, the floor assumes 1,056 tokens per image and at least one image. Count-only units are `images`, `images_1024_low`, `images_1024_medium`, `images_1024_high` and `images_4k`.
- **Per-image diffusion** (DALL-E, Imagen, Titan Image, Nova Canvas, Stable Diffusion, FLUX) uses `images`, with `megapixels=` above 1024x1024 (linear scaling, an assumption). Closed models with no disclosed size are D-large.
- **Transcription.** Whisper and managed services use `audio_minutes` or `audio_seconds`. For gpt-4o-transcribe models pass the reported audio `input_tokens` and text `output_tokens`. Minutes are a fallback that assumes 1,000 audio tokens per minute.

```python
import saef

saef.model_type_for("text-embedding-3-small")                                 # ('embedding_text', 'E-small')
e = saef.estimate_modality("whisper-1", 60, "audio_minutes", region="eu-west-1")
e.energy_wh, e.kg_co2e_low, e.kg_co2e, e.kg_co2e_high, e.water_ml             # facility Wh, kg CO2e, mL
saef.estimate(model="gpt-image-1.5", region="eu-west-1",
              usage={"input_tokens": 50, "image_output_tokens": 4160, "images": 1})
```

```bash
saef modality --model dall-e-3 --unit images --count 100 --region eu-west-1
```

- **Routing.** A billing unit that names a model type (images, audio minutes, search units) decides before the model name. If the name does not fit the unit, the conservative default class of the unit is used and flagged in `warnings`.
- **Energy chain.** Values are stored at GPU level and carried through the text chain unchanged: x 1.5 host x 1.1 idle = server, x 1.2 PUE = facility, then grid and water per region.
- **Lifecycle.** Construction and embodied hardware are charged per server kWh at the class A ratio. **Training is excluded** for every model type in this section, because there is no basis for it. Disclose it as excluded.
- **Ranges.** `low` and `high` are the class range of the energy at central overhead. `kg_co2e_high_case` is the central energy at the conservative overhead, as for text. The two are not combined. Ranges span two orders of magnitude for embeddings and Whisper, so report them with the central value.
- **Excluded.** Text-to-speech, realtime voice, video, music, OCR and fine-tuning have no factor in v3.4.0 and raise an error. Free or failed generations that no export lists are not counted.

## Estimates, not measurements

These figures are modelled estimates built from published benchmarks, grid averages and lifecycle studies. They are not measurements of any provider's infrastructure. The tables carry six significant figures as arithmetic, but the inputs support about two. Report the central value with the high case, and read the methodology before quoting them.

This package bundles the SAEF v3.4.0 tables (text values dated 2026-10-02 and identical to v3.3.0, plus the non-text model types) and covers the H100 central case. Reserved accelerator-hour factors, the B200 sensitivity and the Tier 2b seat-based estimates in the full deposit are not exposed by this API.

## Licence

The dataset is licensed under the SOMA AI Emission Factors Data Licence 1.0 (Non-Commercial), see `LICENSE`, from dataset v3.3.0. The Python code is licensed under PolyForm Noncommercial 1.0.0, see `LICENSE-CODE`. Free for personal use, academic and non-profit research, journalism, charities and government. Any use by a company or freelancer, including internal ESG/CSRD reporting, needs a commercial licence: https://somaai.earth/licensing (guillermollopis@somaai.earth). Versions up to v3.2.0 (package 3.6.0 and earlier) remain under CC BY 4.0 (data) and MIT (code). Attribution for the data is required, see the citation below.

## Citation

Dataset (concept DOI, always the latest version): https://doi.org/10.5281/zenodo.20443585

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

- Homepage: https://somaai.earth/ai-footprint
- Docs: https://somaai.earth/ai-footprint/how-to/track-ai-api-emissions
- Repository: https://github.com/guillermollopis/soma-ai-emission-factors
- Paper: https://doi.org/10.1016/j.spc.2026.11.003 (preprint https://arxiv.org/abs/2606.10660)

mcp-name: io.github.guillermollopis/saef
