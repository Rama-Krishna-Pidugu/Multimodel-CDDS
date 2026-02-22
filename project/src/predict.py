import pickle

# load model & vectorizer (run from project root: python src/predict.py)
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


def predict(disease, drug):
    text = disease.lower() + " " + drug.lower()
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    if pred == 1:
        print("Correct prescription")
    else:
        print("Wrong prescription")


# user input
disease = input("Enter disease: ")
drug = input("Enter drug: ")

predict(disease, drug)
