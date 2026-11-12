# saef

Python access to SAEF, the SOMA AI Emission Factors. SAEF is an open, versioned dataset of estimated carbon and water factors for AI inference, by model class and region, built for Scope 3 Category 1 reporting under the GHG Protocol and CSRD. This package bundles the CSV files and gives you a small API and command line on top of them. It has no dependencies beyond the standard library.

Every number comes with the file, column and row it was read from, so you can trace an estimate back to the dataset.

## Install

```bash
pip install saef
```

For pandas users, `pip install "saef[pandas]"` adds the optional extra.

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
print(len(saef.regions()), saef.__version__)      # 143 cloud regions, 3.0.0
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

This package bundles the SAEF v3.0.0 tables (dataset date 2026-10-02) and covers the H100 central case. Reserved accelerator-hour factors, the B200 sensitivity and the Tier 2b seat-based estimates in the full deposit are not exposed by this API.

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
