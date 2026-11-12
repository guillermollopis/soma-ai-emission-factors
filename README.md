# SOMA AI Emission Factors (SAEF)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20443585.svg)](https://doi.org/10.5281/zenodo.20443585)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10660-b31b1b.svg)](https://arxiv.org/abs/2606.10660)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

SOMA AI Emission Factors (SAEF) is an open, versioned dataset of estimated carbon and water factors for AI inference, by model class and by region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. It covers 210 countries and economies, 27 US eGRID subregions and 143 AWS, Azure and Google Cloud regions. Every factor traces to a file and a source, and the calculator returns the file, line and column behind each number. Published by [SOMA AI](https://somaai.earth), Barcelona, under CC BY 4.0. The method paper (arXiv 2606.10660) is accepted for publication in Sustainable Production and Consumption.

This repository holds the v3.0.0 tables (lifecycle boundary, dataset date 2026-10-02, version DOI [10.5281/zenodo.23371447](https://doi.org/10.5281/zenodo.23371447)) and the source of the `saef` Python package. The complete deposit, with the reference calculator, the generator scripts and the supplementary documents, is on [Zenodo](https://doi.org/10.5281/zenodo.20443585).

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
  version      = {v3.0.0},
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

The data is licensed under CC BY 4.0, see [LICENSE](LICENSE). Attribution: Llopis-Garcia, G. (2026), SOMA AI Emission Factors (SAEF), https://doi.org/10.5281/zenodo.20443585. The Python package code in `src/` is proposed under MIT, see `LICENSE-CODE`.
