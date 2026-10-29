# SOMA AI Emission Factors (SAEF)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20443585.svg)](https://doi.org/10.5281/zenodo.20443585)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10660-b31b1b.svg)](https://arxiv.org/abs/2606.10660)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

SOMA AI Emission Factors (SAEF) is an open, versioned dataset of estimated carbon and water factors for AI inference, per million tokens, by model class and by region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. It covers 210 countries and economies, 27 US eGRID subregions and 143 AWS, Azure and Google Cloud regions. Every factor traces to a file and a source, and the Python calculator returns the file and column behind each number. Published by [SOMA AI](https://somaai.earth), Barcelona, under CC BY 4.0. The method paper, arXiv 2606.10660, is a preprint under peer review.

This repository holds the v2.0.0 snapshot of the CSV files (17 cloud regions). The current release is Zenodo v3.0.0 (2026-10-15, version DOI [10.5281/zenodo.23371447](https://doi.org/10.5281/zenodo.23371447)), which covers 210 countries and economies, 27 US eGRID subregions and 143 cloud regions.

## Files

The files below are the v2.0.0 snapshot kept in this repository. The full v3.0.0 file set is on [Zenodo](https://doi.org/10.5281/zenodo.20443585).

| File | Rows | What it contains |
|---|---|---|
| `carbon_factors.csv` | 53 | Carbon intensity (kg CO₂e per 1M output tokens) by model class × region × infra scenario |
| `water_factors.csv` | 49 | Water consumption (mL per 1,000 output tokens), scope-1 + scope-2, by model class × region |
| `energy_factors.csv` | 9 | Facility-level GPU energy (Wh per 1,000 tokens) by model class and infra scenario |
| `grid_intensity.csv` | 17 | Grid carbon intensity (kg CO₂e/kWh) by cloud region, 2023 location-based averages |
| `energy_per_token_raw.csv` | 19 | Raw GPU energy measurements underlying `energy_factors.csv` |
| `lifecycle_factors.csv` | 7 | Region-independent embodied-hardware and amortised-training add-ons per 1M tokens |
| `model_class_mapping.csv` | 30 | Maps named models to the four model classes, with reasoning-token multipliers |
| `tools/check_factors.py` | — | Recomputes and cross-checks every carbon/water row for internal consistency |
| `README.md` | — | Full column-by-column documentation (this file's counterpart in the dataset itself) |

Four model classes are used throughout: `class_a_small` (~8B), `class_b_mid` (~70B dense), `class_c_large_moe` (frontier MoE), `class_c_large_dense` (frontier dense, ~405B). See `model_class_mapping.csv` to map a specific model name to a class.

## Quick start

```python
import pandas as pd

carbon = pd.read_csv("carbon_factors.csv")

# Or straight from GitHub without cloning:
# carbon = pd.read_csv("https://raw.githubusercontent.com/guillermollopis/soma-ai-emission-factors/main/carbon_factors.csv")

row = carbon[
    (carbon.model_class == "class_b_mid") &
    (carbon.infra_scenario == "H100-central") &
    (carbon.region_id.str.contains("Ireland"))
]
print(row["kg_co2e_per_1m_tokens"].iloc[0])  # 0.04575
```

Inventory formula (Tier 2a):

```
CO2e (kg) = Total_output_tokens_millions × kg_co2e_per_1m_tokens
```

## Carbon factor by region — class B ("GPT-4o / Claude Sonnet" tier), H100-central

Generated from `carbon_factors.csv`, filtered to `model_class == class_b_mid` and `infra_scenario == H100-central`, using the `region_id`, `grid_kg_co2e_per_kwh`, and `kg_co2e_per_1m_tokens` columns, sorted ascending by grid intensity.

| Region | Region ID | Grid intensity (kg CO₂e/kWh) | Carbon factor (kg CO₂e / 1M tokens) |
|---|---|---|---|
| Sweden | `eu-north-1` | 0.0384 | 0.00621 |
| France | `eu-west-3` | 0.0533 | 0.00862 |
| Finland | `europe-north1` | 0.0813 | 0.01314 |
| Belgium | `europe-west1` | 0.135 | 0.02183 |
| UK / London | `europe-west2` | 0.236 | 0.03815 |
| Netherlands | `europe-west4` | 0.268 | 0.04333 |
| US East (N. Virginia) | `us-east-1` | 0.2708 | 0.04378 |
| Ireland | `eu-west-1` | 0.283 | 0.04575 |
| US West (Oregon) | `us-west-2` | 0.2866 | 0.04633 |
| US South Central (Texas) | `us-south-central` | 0.3329 | 0.05382 |
| US South | `us-south` | 0.3821 | 0.06177 |
| Global average | `global-average` | 0.400 | 0.06467 |
| US Central (Iowa) | `us-central1` | 0.4174 | 0.06748 |
| Singapore | `ap-southeast-1` | 0.471 | 0.07615 |
| Japan | `ap-northeast-1` | 0.492 | 0.07954 |

Same model class, same infrastructure, a ~12.3x spread from region alone in the v2.0.0 snapshot (Sweden to Singapore). Region is usually the dominant lever in an inference-heavy inventory, not model choice. See `lifecycle_factors.csv` for the region-independent embodied-hardware and training add-ons to add on top of this electricity term (the "three-term factor" in the dataset README).

## How the checker works

`tools/check_factors.py` is the reproducibility harness run before every deposit:

1. Reads `energy_factors.csv` and confirms the three H100-central facility-energy constants (0.04000 / 0.16167 / 0.20567 Wh per 1,000 tokens) match what's hardcoded as expected.
2. Recomputes every H100-central row in `carbon_factors.csv` as `facility_wh_per_1k_tokens × grid_kg_co2e_per_kwh` and compares it to the published value (tolerance 0.000006 kg/1M tokens), also checking that the redundant `grid_kg_co2e_per_kwh` and `facility_wh_per_1k_tokens` columns match the source tables and that `g_co2e_per_1k_tokens` equals `kg_co2e_per_1m_tokens`.
3. Recomputes every row in `water_factors.csv` from `scope1_wue_l_per_kwh` and `scope2_ewif_l_per_kwh` (tolerance 0.0006 mL/1k tokens) and checks the PUE and facility-energy columns.
4. If a copy of the SOMA product's `aiEmissionFactors.ts` is found (via the `SOMA_PRODUCT_FACTORS` environment variable, or the default relative path), cross-checks every H100-central carbon row for the MoE-calibrated classes, every water row, and both lifecycle terms against the equivalent product constants. If no product file is found, this parity check is skipped with a message rather than failing.

Run it with:

```bash
python tools/check_factors.py
```

Prints `OK` and exits 0 if everything reconciles; otherwise lists each mismatch and exits 1.

## Citation

```bibtex
@misc{llopis2026somafactors,
  author       = {Llopis, Guillem},
  title        = {{SOMA AI Emission Factors (SAEF)}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3.0.0},
  doi          = {10.5281/zenodo.20443585},
  url          = {https://doi.org/10.5281/zenodo.20443585}
}

@misc{llopis2026accounting,
  author        = {Llopis, Guillem},
  title         = {Accounting for Purchased Artificial Intelligence Inference in Corporate Greenhouse Gas Inventories: A Token-Based Method},
  year          = {2026},
  eprint        = {2606.10660},
  archivePrefix = {arXiv},
  note          = {Under peer review at Sustainable Production and Consumption (Elsevier)}
}
```

Full derivations, all columns, boundary conditions, and the changelog are documented in the dataset's own `README.md` (deposited alongside the CSVs on Zenodo and Hugging Face).

## Links

- Zenodo (concept DOI, always latest version): https://doi.org/10.5281/zenodo.20443585
- Zenodo v3.0.0 record: https://zenodo.org/records/23371447
- Website: https://somaai.earth/ai-footprint/factors
- Paper (arXiv, preprint under peer review): https://arxiv.org/abs/2606.10660
- Methodology: https://app.somaai.earth/methodology
- Developer docs / API: https://app.somaai.earth/docs/ai-factors
- Factor pages: https://app.somaai.earth/ai-factors
- Contact: guillermollopis@somaai.earth

## License

CC BY 4.0 — see [LICENSE](LICENSE). Attribution: Llopis, G. (2026), SOMA AI Emission Factors (SAEF), https://doi.org/10.5281/zenodo.20443585.
