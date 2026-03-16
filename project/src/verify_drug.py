"""
Drug verification layer: checks against trusted clinical source.
Replace this module later with real-time API (e.g. MedlinePlus) without changing callers.
"""
import json
import os

# Load from project root when running as: python src/predict.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_db_path = os.path.join(_project_root, "data", "trusted_drug_db.json")

with open(_db_path, encoding="utf-8") as f:
    trusted_db = json.load(f)


def verify_with_trusted_source(disease, drug):
    disease = disease.lower()
    drug = drug.lower()

    if disease in trusted_db:
        for valid_drug in trusted_db[disease]:
            if valid_drug in drug:
                return True

    return False
