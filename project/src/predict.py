import os
import pickle
import sys

# Allow importing from src when run as: python src/predict.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prescription_parser import extract_drugs_from_text
from verify_drug import verify_with_trusted_source

# load model & vectorizer (run from project root: python src/predict.py)
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


def predict(disease, drug):
    text = disease.lower() + " " + drug.lower()
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred


disease = input("Enter disease: ")
print("Paste full prescription text (multiple lines allowed).")
print("Example: 'Tab Metformin 500mg BD, Tab Amlodipine 5mg OD'")
prescription_text = sys.stdin.read()

drugs = extract_drugs_from_text(prescription_text)

if not drugs:
    print("\nNo drugs detected in prescription text.")
    sys.exit(0)

all_appropriate = True
all_verified = True

print("\nPrescription analysis:\n")

for drug in drugs:
    ml_result = predict(disease, drug)
    trusted_result = verify_with_trusted_source(disease, drug)

    display_name = drug

    if ml_result == 1:
        if trusted_result:
            print(f"{display_name} -> Appropriate for {disease}")
            print("  Verified from trusted clinical source")
        else:
            print(f"{display_name} -> Appropriate (model)")
            print("  Not found in trusted source")
            all_verified = False
    else:
        print(f"{display_name} -> Not appropriate for {disease}")
        if trusted_result:
            print("  Found in trusted source - consider clinical review")
        all_appropriate = False

if all_appropriate and all_verified:
    print("\nFINAL VERDICT: Prescription is appropriate and verified from trusted source")
elif all_appropriate:
    print("\nFINAL VERDICT: Prescription is appropriate (model); needs clinical review for unverified drugs")
else:
    print("\nFINAL VERDICT: Prescription contains inappropriate drug(s)")

