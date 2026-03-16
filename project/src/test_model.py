import pickle

# load model & vectorizer (run from project root: python src/test_model.py)
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


def predict(disease, drug):
    text = disease.lower() + " " + drug.lower()
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


print("\nTEST RESULTS:\n")

with open("test_cases.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        disease, drug = line.split("|")

        disease = disease.strip()
        drug = drug.strip()

        result = predict(disease, drug)

        if result == 1:
            verdict = "Appropriate"
        else:
            verdict = "Not appropriate"

        print(f"{disease} + {drug} -> {verdict}")
