"""Utilities for text normalization and disease keyword mapping."""

from __future__ import annotations

import re
from typing import Dict, List

TARGET_DISEASES: List[str] = [
    "pneumonia",
    "tuberculosis",
    "diabetes",
    "hypertension",
]

# First matching disease wins if multiple keyword groups are present.
DISEASE_KEYWORDS: Dict[str, List[str]] = {
    "pneumonia": [
        "pneumonia",
        "lower respiratory tract infection",
        "respiratory tract infection",
        "chest infection",
    ],
    "tuberculosis": [
        "tuberculosis",
        " tb ",
        "tb ",
        " tb",
    ],
    "diabetes": [
        "diabetes",
        "type 2 diabetes",
        "hyperglycemia",
        "high blood sugar",
    ],
    "hypertension": [
        "hypertension",
        "high blood pressure",
    ],
}


def normalize_text(text: str) -> str:
    """Lowercase and collapse punctuation/spacing noise."""
    if text is None:
        return ""

    cleaned = str(text).lower()
    cleaned = cleaned.replace("/", " ")
    cleaned = re.sub(r"[^a-z0-9+\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Standardization aliases
    cleaned = cleaned.replace("type ii", "type 2")
    cleaned = cleaned.replace("high bp", "high blood pressure")

    return cleaned


def canonicalize_drug_name(drug_name: str) -> str:
    """Canonical form for medicine names before feature building."""
    text = normalize_text(drug_name)
    text = re.sub(r"\b\d+\s*mg\b", "", text)
    text = re.sub(r"\b(tablet|capsule|injection|syrup|cream|gel|drops|solution)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_respiratory_context(normalized_uses: str) -> bool:
    respiratory_markers = [
        "respiratory",
        "lung",
        "chest",
        "pulmonary",
    ]
    return any(marker in normalized_uses for marker in respiratory_markers)


def extract_target_diseases_from_uses(uses_text: str) -> List[str]:
    """
    Map a free-text Uses field into one or more target diseases.

    Notes:
    - "bacterial infection" should only map to pneumonia if respiratory context exists.
    """
    normalized = f" {normalize_text(uses_text)} "
    hits: List[str] = []

    for disease, keywords in DISEASE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            hits.append(disease)

    if "bacterial infection" in normalized and _contains_respiratory_context(normalized):
        if "pneumonia" not in hits:
            hits.append("pneumonia")

    # Stable order and dedupe
    ordered_hits = [d for d in TARGET_DISEASES if d in set(hits)]
    return ordered_hits


def build_feature_text(disease: str, drug_name: str) -> str:
    """Create the joint text field used by TF-IDF."""
    return f"{normalize_text(disease)} [SEP] {canonicalize_drug_name(drug_name)}"
