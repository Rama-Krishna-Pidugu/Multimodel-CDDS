# Phase-1: Disease-Drug Prescription Validator (Binary)

This project implements a text-based Clinical Decision Support baseline that predicts whether a prescribed drug is appropriate for a given disease.

## Scope
- Phase-1 only (text input).
- Diseases: `pneumonia`, `tuberculosis`, `diabetes`, `hypertension`.
- Output labels: `correct` (1) and `incorrect` (0).
- X-ray and OCR are intentionally deferred to later phases.

## Dataset
Source file expected at:
- `data/Medicine_Details.csv`

Required source columns:
- `Medicine Name`
- `Uses`

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Build Processed Pairs
```bash
python src/dataset_builder.py --input data/Medicine_Details.csv --output data/processed_pairs.csv
```

Output schema (`data/processed_pairs.csv`):
- `disease` (string)
- `drug_name` (string)
- `label` (int: 1 correct, 0 incorrect)
- `source_type` (`positive|negative_generated`)

## Train Model
```bash
python src/train.py --data data/processed_pairs.csv --model_out models/model.joblib --metrics_out models/metrics.json
```

Artifacts:
- `models/model.joblib`
- `models/metrics.json`
- `models/confusion_matrix.png`

## Predict from CLI
```bash
python src/predict.py --disease "diabetes" --prescription "metformin, amoxicillin"
```

Decision rule:
- `APPROPRIATE` if all drugs are predicted `correct`.
- `INAPPROPRIATE` otherwise.

## Modeling Baseline
- Feature text: `"{disease} [SEP] {drug_name}"`
- `TfidfVectorizer(ngram_range=(1,2), min_df=2)`
- `LogisticRegression(max_iter=1000, class_weight='balanced')`
- Split: stratified 80/20 with seed `42`

## Validation Checks Included
- Missing required columns fail with explicit error.
- Empty `Uses` rows are skipped and counted.
- Duplicate `(disease, drug)` pairs are removed.
- Prediction supports case-insensitive and extra-space inputs.

## Limitations
- Controlled disease vocabulary only.
- Static dataset-derived knowledge.
- Binary labeling only in this phase.

## Future Work
- Phase-2: X-ray disease prediction integration.
- Phase-3: OCR from printed/handwritten prescription images.
- Optional dynamic web knowledge augmentation for unseen pairs.
