"""Regenerate data/saef-data.json from the Python package's bundled CSVs.

Usage: python scripts/build-data.py [path/to/saef-repo/src/saef]
"""
import csv
import io
import json
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Guillem/AppData/Local/Temp/saef-repo/src/saef")
data_dir = src / "data"
out_path = Path(__file__).resolve().parent.parent / "data" / "saef-data.json"


def rows(name):
    text = (data_dir / name).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def pick(rs, cols):
    return [{c: r[c] for c in cols} for r in rs]


class_cols = []
for c in "abc":
    class_cols += [
        f"class_{c}_inference_footprint_kg_per_m_effective_tokens",
        f"class_{c}_training_kg_per_m_billed_tokens",
        f"class_{c}_inference_footprint_high_case_kg_per_m_effective_tokens",
        f"class_{c}_development_line_kg_per_m_effective_tokens",
        f"class_{c}_electricity_kg_per_m_effective_tokens",
        f"class_{c}_embodied_kg_per_m_effective_tokens",
        f"class_{c}_construction_kg_per_m_effective_tokens",
        f"class_{c}_factor_kg_per_m_billed_tokens_default_mix",
    ]

version = "3.1.0"
init = (src / "_core.py").read_text(encoding="utf-8")
assert f'DATASET_VERSION = "{version}"' in init

out = {
    "datasetVersion": version,
    "datasetDoi": "10.5281/zenodo.20443585",
    "datasetVersionDoi": "10.5281/zenodo.24160409",
    "paperDoi": "10.1016/j.spc.2026.11.003",
    "constants": pick(rows("token_basis_constants.csv"), ["constant", "central"]),
    "modelClassMapping": pick(
        rows("model_class_mapping.csv"),
        ["pattern", "provider", "model_class", "basis", "size_or_tier_note", "own_factor_model_id", "notes"],
    ),
    "modelFactors": pick(
        rows("model_factors.csv"),
        [
            "model_id", "model_class", "gpu", "facility_wh_per_1k_effective_tokens",
            "facility_wh_high_case_per_1k_effective_tokens", "embodied_kg_co2e_per_m_tokens",
            "construction_kg_co2e_per_m_tokens", "embodied_high_case_kg_co2e_per_m_tokens",
            "construction_high_case_kg_co2e_per_m_tokens", "training_kg_co2e_per_m_billed_tokens",
            "development_line_kg_co2e_per_m_tokens_global", "grid_kg_co2e_per_kwh",
        ],
    ),
    "cloudRegions": pick(
        rows("cloud_region_factors.csv"),
        ["provider", "region_id", "grid_id", "mapping_confidence", "mapping_notes"] + class_cols,
    ),
    "grids": pick(rows("grid_intensity_all.csv"), ["grid_id", "name", "kind", "grid_kg_co2e_per_kwh"] + class_cols),
    "crosswalk": pick(rows("region_grid_crosswalk.csv"), ["region_id", "grid_id"]),
    "waterFactors": pick(
        rows("region_water_factors.csv"),
        [
            "region_id", "wue_l_per_kwh", "wue_denominator", "wue_basis_as_published",
            "withdrawal_to_consumption", "ewif_l_per_kwh", "ewif_excl_hydro_l_per_kwh", "ewif_source",
        ],
    ),
    "lifecycle": pick(
        [r for r in rows("factors_v3_lifecycle.csv") if r["gpu"] == "H100"],
        [
            "model_class", "gpu", "region_id", "facility_wh_per_1k_effective_tokens",
            "server_wh_per_1k_effective_tokens",
        ],
    ),
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
print(out_path, out_path.stat().st_size)
