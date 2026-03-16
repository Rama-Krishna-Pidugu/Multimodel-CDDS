import re
from typing import List


def _clean_drug_text(text: str) -> str:
    """
    Apply the same cleaning logic used in load_data.py:
    - lowercase
    - remove dosage in parentheses
    - remove digits
    - collapse extra spaces
    """
    text = text.lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_drugs_from_text(prescription_text: str) -> List[str]:
    """
    Extract cleaned drug strings from raw prescription text.

    Strategy:
    - split on commas, newlines, semicolons
    - clean each chunk using the same rules as training
    - drop empty and duplicate entries
    """
    if not prescription_text:
        return []

    raw_chunks = re.split(r"[,\n;]+", prescription_text)
    seen = set()
    drugs: List[str] = []

    for chunk in raw_chunks:
        cleaned = _clean_drug_text(chunk)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            drugs.append(cleaned)

    return drugs

