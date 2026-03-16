from __future__ import annotations

import os
import pickle
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from .verify_drug import verify_with_trusted_source
except ImportError:
    # Fallback when running this file directly.
    from verify_drug import verify_with_trusted_source


app = FastAPI(title="Medical Prescription Validation API", version="1.0.0")

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path = os.path.join(_project_root, "models", "model.pkl")
_vectorizer_path = os.path.join(_project_root, "models", "vectorizer.pkl")

with open(_model_path, "rb") as f:
    model = pickle.load(f)

with open(_vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)


class ValidateRequest(BaseModel):
    disease: str = Field(..., min_length=1)
    drugs: List[str] = Field(..., min_length=1)


class DrugAnalysis(BaseModel):
    drug: str
    prediction: str
    verified_from_trusted_source: bool
    confidence: Optional[float] = None


class ValidateResponse(BaseModel):
    disease: str
    analysis: List[DrugAnalysis]
    final_verdict: str


def _predict_single(disease: str, drug: str) -> tuple[int, Optional[float]]:
    text = f"{disease.lower()} {drug.lower()}"
    vec = vectorizer.transform([text])
    pred = int(model.predict(vec)[0])

    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        confidence = round(float(max(probs)) * 100.0, 2)

    return pred, confidence


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Medical Prescription Validation API is running."}


@app.post("/validate", response_model=ValidateResponse)
def validate_prescription(payload: ValidateRequest) -> ValidateResponse:
    disease = payload.disease.strip()
    drugs = [drug.strip() for drug in payload.drugs if drug.strip()]

    analysis: List[DrugAnalysis] = []
    all_appropriate = True
    all_verified = True

    for drug in drugs:
        pred, confidence = _predict_single(disease, drug)
        trusted_result = verify_with_trusted_source(disease, drug)

        if pred == 1:
            prediction = "Appropriate"
            if not trusted_result:
                all_verified = False
        else:
            prediction = "Not Appropriate"
            all_appropriate = False

        analysis.append(
            DrugAnalysis(
                drug=drug,
                prediction=prediction,
                verified_from_trusted_source=trusted_result,
                confidence=confidence,
            )
        )

    if all_appropriate and all_verified:
        final_verdict = (
            "Final Verdict: Prescription is appropriate and verified from trusted source"
        )
    elif all_appropriate:
        final_verdict = (
            "Final Verdict: Prescription is appropriate (model); needs clinical review for unverified drugs"
        )
    else:
        final_verdict = "Final Verdict: Prescription contains inappropriate drug(s)"

    return ValidateResponse(disease=disease, analysis=analysis, final_verdict=final_verdict)
