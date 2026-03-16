import os
import pickle
from typing import List

import easyocr
import streamlit as st

from src.prescription_parser import extract_drugs_from_text
from src.verify_drug import verify_with_trusted_source


@st.cache_resource
def load_model_and_vectorizer():
    model_path = os.path.join("models", "model.pkl")
    vec_path = os.path.join("models", "vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"])


def predict_appropriateness(model, vectorizer, disease: str, drug: str) -> int:
    text = f"{disease.lower()} {drug.lower()}"
    vec = vectorizer.transform([text])
    return int(model.predict(vec)[0])


def run_ocr_on_image(uploaded_file) -> List[str]:
    reader = get_ocr_reader()
    image_bytes = uploaded_file.read()
    # easyocr can take a bytes-like object directly
    results = reader.readtext(image_bytes, detail=0)
    return results


def main():
    st.set_page_config(page_title="AI Prescription Validator", layout="centered")

    st.title("AI Prescription Validator")
    st.write("Phase‑3 demo: OCR + drug appropriateness check")

    model, vectorizer = load_model_and_vectorizer()

    disease = st.selectbox("Select disease", ["diabetes", "hypertension"])

    uploaded_image = st.file_uploader(
        "Upload prescription image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None and st.button("Analyze prescription"):
        with st.spinner("Running OCR and analysis..."):
            ocr_lines = run_ocr_on_image(uploaded_image)
            full_text = "\n".join(ocr_lines)
            drugs = extract_drugs_from_text(full_text)

        st.subheader("Raw OCR text")
        st.code(full_text or "(no text detected)")

        if not drugs:
            st.warning("No drug candidates detected after cleaning.")
            return

        st.subheader("Drug analysis")

        rows = []
        all_appropriate = True
        all_verified = True

        for drug in drugs:
            ml_result = predict_appropriateness(model, vectorizer, disease, drug)
            trusted = verify_with_trusted_source(disease, drug)

            if ml_result == 1:
                if trusted:
                    status = "Appropriate (verified)"
                else:
                    status = "Appropriate (model only)"
                    all_verified = False
            else:
                status = "Not appropriate"
                all_appropriate = False

            rows.append(
                {
                    "Drug": drug,
                    "ML decision": "appropriate" if ml_result == 1 else "not appropriate",
                    "Trusted source": "yes" if trusted else "no",
                    "Summary": status,
                }
            )

        st.table(rows)

        if all_appropriate and all_verified:
            st.success("FINAL VERDICT: Prescription is appropriate and verified from trusted source.")
        elif all_appropriate:
            st.info(
                "FINAL VERDICT: Prescription is appropriate by the model; some drugs are not in the trusted list and need clinical review."
            )
        else:
            st.error("FINAL VERDICT: Prescription contains inappropriate drug(s).")


if __name__ == "__main__":
    main()

