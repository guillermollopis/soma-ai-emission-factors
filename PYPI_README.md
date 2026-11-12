# saef

Python access to SAEF, the SOMA AI Emission Factors. SAEF is an open, versioned dataset of estimated carbon and water factors for AI inference, per million tokens, by model class and region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. This package bundles the CSV files and gives you a small API and command line on top of them. It has no dependencies beyond the standard library.

Every number comes with the file, column and row it was read from, so you can trace an estimate back to the dataset.

## Install

```bash
pip install saef
```

For pandas users, `pip install "saef[pandas]"` adds the optional extra.

## Example

```python
import saef

print(saef.model_class_for("gpt-4o"))            # class_b_mid
print(saef.factor("B", "eu-west-1")["kg_co2e_per_1m_tokens"])

r = saef.estimate(tokens=1_000_000, model="gpt-4o-mini", region="eu-west-1")
print(r.kg_co2e, r.water_ml)                      # kg CO2e and mL of water
print(r.sources["electricity"])                   # file, column and row behind the number
print(saef.regions()[:3], saef.__version__)
print(saef.citation())
```

Command line:

```bash
saef estimate --tokens 1000000 --model gpt-4o-mini --region eu-west-1
saef factor --model-class B --region us-east-1
saef regions
saef cite
```

## What the numbers are

The total factor is the sum of three terms, all in kg CO2e per million output tokens:

- Electricity, from `carbon_factors.csv` (facility energy multiplied by grid intensity).
- Embodied hardware, from `lifecycle_factors.csv`.
- Amortised training, from `lifecycle_factors.csv`.

Water, in mL, comes from `water_factors.csv` (scope 1 and scope 2) where a row exists for the class and region. Models with hidden reasoning tokens carry a multiplier from `model_class_mapping.csv`, which `estimate` applies to your token count. Pass `reasoning_multiplier=1` to switch it off.

Regions can be given as an id (`eu-west-1`), a place name (`ireland`) or an unambiguous prefix (`us-east`). An ambiguous or unknown region raises `saef.SaefError` with the candidates listed.

## Estimates, not measurements

These figures are modelled estimates built from published benchmarks, grid averages and lifecycle studies. They are not measurements of any provider's actual infrastructure. Uncertainty is large, and the lifecycle terms are skewed high. Use them for inventory screening and Scope 3 reporting, and read the methodology before quoting them.

The bundled data is the SAEF v2.0.0 snapshot with 17 cloud regions. The full v3.0.0 release on Zenodo covers more regions and countries.

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
