"""Copy the dataset CSVs from the repo root into src/saef/data so the package ships them.
The repo-root CSVs stay the source of truth. Run before every build: python scripts/sync_data.py"""
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "saef" / "data"
FILES = [
    "carbon_factors.csv",
    "water_factors.csv",
    "energy_factors.csv",
    "grid_intensity.csv",
    "lifecycle_factors.csv",
    "model_class_mapping.csv",
    "energy_per_token_raw.csv",
]

DEST.mkdir(parents=True, exist_ok=True)
for name in FILES:
    shutil.copyfile(ROOT / name, DEST / name)
    print(f"synced {name}")
