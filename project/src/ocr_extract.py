import os
import sys
from typing import List

import easyocr

# Allow importing from src when run as: python src/ocr_extract.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prescription_parser import extract_drugs_from_text


def extract_text_from_image(image_path: str) -> List[str]:
    reader = easyocr.Reader(["en"])
    results = reader.readtext(image_path, detail=0)
    return results


def main() -> None:
    image_path = input("Enter prescription image path: ").strip()
    if not image_path:
        print("No image path provided.")
        return

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("\nRunning OCR...")
    lines = extract_text_from_image(image_path)

    if not lines:
        print("No text detected in image.")
        return

    full_text = "\n".join(lines)
    print("\nRaw OCR text:\n")
    print(full_text)

    drugs = extract_drugs_from_text(full_text)

    print("\nExtracted drug candidates (cleaned):\n")
    for d in drugs:
        print(f"- {d}")


if __name__ == "__main__":
    main()

