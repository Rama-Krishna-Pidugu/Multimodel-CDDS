import pandas as pd

# load cleaned dataset (run from project root: python src/extract_diseases.py)
df = pd.read_csv("data/clean_medicine_data.csv")

# lowercase for matching
df["uses"] = df["uses"].str.lower()

# disease mapping
disease_map = {
    "diabetes": ["diabetes"],
    "hypertension": ["hypertension", "high blood pressure"],
}

rows = []

for _, row in df.iterrows():
    for disease, keywords in disease_map.items():
        if any(keyword in str(row["uses"]) for keyword in keywords):
            rows.append([row["drug"], disease])

# create new dataframe
disease_df = pd.DataFrame(rows, columns=["drug", "disease"])

# remove duplicates
disease_df = disease_df.drop_duplicates()

# save it
disease_df.to_csv("data/disease_drug_positive.csv", index=False)

print("Extracted disease-drug pairs")
print(disease_df.head())
print("\nTotal pairs:", len(disease_df))

