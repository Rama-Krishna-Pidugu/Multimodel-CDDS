"""Build processed disease-drug pairs from the medicine details CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from preprocessing import (
    TARGET_DISEASES,
    canonicalize_drug_name,
    extract_target_diseases_from_uses,
)

REQUIRED_COLUMNS = ["Medicine Name", "Uses"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed training pairs")
    parser.add_argument(
        "--input",
        default="data/Medicine_Details.csv",
        help="Path to source CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed_pairs.csv",
        help="Path to write processed pairs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Expected columns include: {REQUIRED_COLUMNS}"
        )


def find_existing_input(path_str: str) -> Path:
    candidates = [Path(path_str), Path("src") / "data" / Path(path_str).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Input CSV not found. Tried: {[str(c) for c in candidates]}")


def build_positive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    positive_rows: List[dict] = []
    skipped_empty_uses = 0

    for _, row in df.iterrows():
        raw_uses = row.get("Uses")
        if pd.isna(raw_uses) or not str(raw_uses).strip():
            skipped_empty_uses += 1
            continue

        diseases = extract_target_diseases_from_uses(str(raw_uses))
        if not diseases:
            continue

        drug_name = canonicalize_drug_name(str(row["Medicine Name"]))
        if not drug_name:
            continue

        for disease in diseases:
            positive_rows.append(
                {
                    "disease": disease,
                    "drug_name": drug_name,
                    "label": 1,
                    "source_type": "positive",
                }
            )

    positives = pd.DataFrame(positive_rows)
    if positives.empty:
        raise ValueError("No positive pairs were extracted. Check disease keyword mapping.")

    positives = positives.drop_duplicates(subset=["disease", "drug_name"])
    print(f"[info] Skipped rows with empty Uses: {skipped_empty_uses}")
    print(f"[info] Positive pairs after dedupe: {len(positives)}")

    return positives


def build_negative_pairs(positives: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    negative_rows: List[dict] = []
    positives_by_disease = positives.groupby("disease")

    for disease in TARGET_DISEASES:
        disease_pos = positives_by_disease.get_group(disease) if disease in positives_by_disease.groups else pd.DataFrame()
        if disease_pos.empty:
            continue

        disease_pos_count = len(disease_pos)

        other_diseases = [d for d in TARGET_DISEASES if d != disease]
        candidate_drugs = positives[positives["disease"].isin(other_diseases)]["drug_name"].drop_duplicates().tolist()

        if not candidate_drugs:
            continue

        candidate_pairs = pd.DataFrame(
            {
                "disease": disease,
                "drug_name": candidate_drugs,
                "label": 0,
                "source_type": "negative_generated",
            }
        )

        candidate_pairs = candidate_pairs.merge(
            disease_pos[["disease", "drug_name"]],
            on=["disease", "drug_name"],
            how="left",
            indicator=True,
        )
        candidate_pairs = candidate_pairs[candidate_pairs["_merge"] == "left_only"].drop(columns=["_merge"])

        max_neg = min(len(candidate_pairs), disease_pos_count * 2)
        if max_neg > 0:
            sampled_idx = rng.choice(candidate_pairs.index.to_numpy(), size=max_neg, replace=False)
            sampled = candidate_pairs.loc[sampled_idx]
            negative_rows.extend(sampled.to_dict(orient="records"))

    negatives = pd.DataFrame(negative_rows)
    if negatives.empty:
        raise ValueError("No negative pairs were generated. Check positive extraction and disease coverage.")

    negatives = negatives.drop_duplicates(subset=["disease", "drug_name"])
    print(f"[info] Negative pairs after balancing+dedupe: {len(negatives)}")

    return negatives


def main() -> None:
    args = parse_args()

    input_path = find_existing_input(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_schema(df)

    positives = build_positive_pairs(df)
    negatives = build_negative_pairs(positives, seed=args.seed)

    processed = pd.concat([positives, negatives], ignore_index=True)
    processed = processed.drop_duplicates(subset=["disease", "drug_name", "label"])

    processed.to_csv(output_path, index=False)

    print("[done] Processed dataset written")
    print(f"[done] Input : {input_path}")
    print(f"[done] Output: {output_path}")
    print("[done] Label distribution:")
    print(processed["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
