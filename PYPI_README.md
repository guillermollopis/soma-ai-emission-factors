# saef

Python access to SAEF, the SOMA AI Emission Factors. SAEF is an open, versioned dataset of estimated carbon and water factors for AI inference, by model class and region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. This package bundles the CSV files and gives you a small API and command line on top of them. It has no dependencies beyond the standard library.

Every number comes with the file, column and row it was read from, so you can trace an estimate back to the dataset.

## Install

```bash
pip install saef
```

For pandas users, `pip install "saef[pandas]"` adds the optional extra. The framework integrations below use `saef[litellm]` and `saef[langchain]`.

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
print(len(saef.regions()), saef.__version__, saef.DATASET_VERSION)  # 143 cloud regions, 3.2.0, 3.1.0
print(saef.citation())
```

Command line:

```bash
saef estimate --tokens 1000000 --model gpt-4o-mini --region eu-west-1
saef estimate --output-tokens 2e6 --input-tokens 10e6 --model gpt-4o --provider azure --region swedencentral
saef factor --model-class B --region us-east-1
saef regions --provider aws
saef cite
```

## Integrations

The core package has no dependencies. The integrations below estimate carbon and water for each call from the token usage your framework already reports. The region comes from the `region` argument, then from the `SAEF_REGION` environment variable, then falls back to the global average. All of them keep running totals and can append one JSON line per call with `log_path=`.

LiteLLM, installed with `pip install "saef[litellm]"`. The estimate is also attached to `response._hidden_params["saef"]`.

```python
import litellm
from saef.integrations.litellm import SaefLogger

logger = SaefLogger(region="eu-west-1")
litellm.callbacks = [logger]
resp = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}], mock_response="Hi there")
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

Every figure is an estimate. Models the dataset treats as out of scope, such as embeddings, are skipped and counted in `totals()["skipped_calls"]`.

## What the numbers are

The factors follow the v3 lifecycle boundary. An estimate has four terms: electricity, embodied hardware and data-centre construction (charged on effective tokens) and amortised training (charged on billed tokens).

- Effective tokens are output tokens plus weighted input tokens, divided by a reference-mix normaliser. Pass `tokens` for a billed total, which applies the default input share of 85% and no cache credit. Pass `output_tokens`, `input_tokens` and `cached_input_tokens` when your billing export gives the split.
- Never multiply a billed token count by a per-effective-token footprint yourself. Use `estimate`, or the `kg_co2e_per_1m_billed_tokens` value from `factor`.
- A model maps to class A, B or C through 187 identifier patterns, including AWS Bedrock, Azure and Vertex AI spellings. Closed models are class proxies by pricing tier, and the result says so in `warnings`. An unmapped identifier is assigned the conservative class C and flagged. Embeddings, image, audio and video models are out of scope and raise an error.
- `region` can be a cloud region id (143 AWS, Azure and Google Cloud regions), an ISO-3 country code (210 countries and economies), a US eGRID subregion, `eu-27`, or `None` for the global average. A region that matches nothing raises `saef.SaefError` with suggestions.
- Water covers 17 regions, under two hydropower conventions (`water_ml` and `water_ml_excl_hydro`). Elsewhere `water_ml` is `None`, by design.
- `kg_co2e_high_case` is a conservative bound. The development line is reported separately and is never added to `kg_co2e`.

## Estimates, not measurements

These figures are modelled estimates built from published benchmarks, grid averages and lifecycle studies. They are not measurements of any provider's infrastructure. The tables carry six significant figures as arithmetic, but the inputs support about two. Report the central value with the high case, and read the methodology before quoting them.

This package bundles the SAEF v3.1.0 tables (dataset date 2026-10-02, a corrective release of v3.0.0) and covers the H100 central case. Reserved accelerator-hour factors, the B200 sensitivity and the Tier 2b seat-based estimates in the full deposit are not exposed by this API.

## Licence

The dataset is licensed under CC BY 4.0. The Python code is licensed under MIT. Attribution for the data is required, see the citation below.

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

Method paper: Llopis-Garcia, G. (2026), Accounting for Purchased Artificial Intelligence Inference in Corporate Greenhouse Gas Inventories, a Token-Based Method. arXiv 2606.10660, accepted for publication in Sustainable Production and Consumption.

## Links

- Homepage: https://somaai.earth/ai-footprint
- Docs: https://somaai.earth/ai-footprint/how-to/track-ai-api-emissions
- Repository: https://github.com/guillermollopis/soma-ai-emission-factors
- Paper: https://arxiv.org/abs/2606.10660
