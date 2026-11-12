# SOMA AI Emission Factors (SAEF)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20443585.svg)](https://doi.org/10.5281/zenodo.20443585)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10660-b31b1b.svg)](https://arxiv.org/abs/2606.10660)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

SOMA AI Emission Factors (SAEF) is an open, versioned dataset of estimated carbon and water factors for AI inference, by model class and by region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. It covers 210 countries and economies, 27 US eGRID subregions and 143 AWS, Azure and Google Cloud regions. Every factor traces to a file and a source, and the calculator returns the file, line and column behind each number. Published by [SOMA AI](https://somaai.earth), Barcelona, under CC BY 4.0. The method paper (arXiv 2606.10660) is accepted for publication in Sustainable Production and Consumption.

This repository holds the v3.1.0 tables (lifecycle boundary, dataset date 2026-10-02, version DOI [10.5281/zenodo.24160409](https://doi.org/10.5281/zenodo.24160409); v3.1.0 is a corrective release of v3.0.0 that maps Azure canadaeast and Google Cloud northamerica-northeast1 to the Quebec grid and replaces the Central African Republic's rounded-to-zero grid row, see the release notes) and the source of the `saef` Python package. The complete deposit, with the reference calculator, the generator scripts and the supplementary documents, is on [Zenodo](https://doi.org/10.5281/zenodo.20443585).

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

The package bundles the nine CSV files below and has no dependencies. It is built from this repository with `python scripts/sync_data.py && python -m build`.

## Integrations

The core package has no dependencies. The integrations below estimate carbon and water for each call from the token usage your framework already reports. The region comes from the `region` argument, then from the `SAEF_REGION` environment variable, then falls back to the global average. All three keep running totals and can append one JSON line per call with `log_path=`.

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

## Files

| File | Rows | What it contains |
|---|---|---|
| `cloud_region_factors.csv` | 143 | Class A, B and C factors for each AWS, Azure and Google Cloud region, resolved to a grid, with the confidence of the region mapping |
| `grid_intensity_all.csv` | 240 | Grid carbon intensity (kg CO2e per kWh) for 210 countries and economies, 27 US eGRID subregions, EU-27, the world average and one subnational row, with the class factors at each grid |
| `factors_v3_lifecycle.csv` | 102 | Class factors by class, GPU (H100 central, B200 sensitivity) and printed region, with the energy and the four terms |
| `model_factors.csv` | 32 | Factors for the 18 measured open-weight models on H100 and B200, from the ML.ENERGY Leaderboard v3 |
| `model_class_mapping.csv` | 187 | Identifier patterns, including AWS Bedrock, Azure and Vertex AI spellings, mapped to class A, B or C, plus the class C default for unmapped names |
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

- An unmapped identifier is assigned class C and flagged. Embeddings, image, audio, video and fine-tuning jobs are out of scope.
- Unknown or global regions use the global-average grid. Water has no fallback, and covers 17 regions only.
- Each result carries a conservative high case. A development line is reported separately and never added to a total.
- Six significant figures are arithmetic, not evidence. The inputs support about two. Report the central value with the high case.

The numbers are estimates, not measurements.

## Citation

```bibtex
@misc{llopis2026somafactors,
  author       = {Llopis-Garcia, Guillermo},
  title        = {{SOMA AI Emission Factors (SAEF)}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3.1.0},
  doi          = {10.5281/zenodo.20443585},
  url          = {https://doi.org/10.5281/zenodo.20443585}
}

@misc{llopis2026accounting,
  author        = {Llopis-Garcia, Guillermo},
  title         = {Accounting for Purchased Artificial Intelligence Inference in Corporate Greenhouse Gas Inventories: A Token-Based Method},
  year          = {2026},
  eprint        = {2606.10660},
  archivePrefix = {arXiv},
  note          = {Accepted for publication in Sustainable Production and Consumption (Elsevier)}
}
```

Pin a dataset in an inventory by the version DOI. The concept DOI always resolves to the latest version.

## Links

- Zenodo (concept DOI, always latest version): https://doi.org/10.5281/zenodo.20443585
- Zenodo v3.1.0 record: https://zenodo.org/records/24160409
- Zenodo v3.0.0 record: https://zenodo.org/records/23371447
- Website: https://somaai.earth/ai-footprint/factors
- Paper (arXiv preprint, accepted for publication in Sustainable Production and Consumption): https://arxiv.org/abs/2606.10660
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

The data is licensed under CC BY 4.0, see [LICENSE](LICENSE). Attribution: Llopis-Garcia, G. (2026), SOMA AI Emission Factors (SAEF), https://doi.org/10.5281/zenodo.20443585. The Python package code in `src/` is licensed under MIT, see `LICENSE-CODE`.
