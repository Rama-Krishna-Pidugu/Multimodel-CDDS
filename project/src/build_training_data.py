import pandas as pd
import random

# load positive disease-drug pairs (run from project root: python src/build_training_data.py)
df = pd.read_csv("data/disease_drug_positive.csv")

# positive samples
df["label"] = 1

# get unique lists
all_drugs = df["drug"].unique()
all_diseases = df["disease"].unique()

negative_rows = []

for _, row in df.iterrows():
    disease = row["disease"]

    # pick a random wrong drug
    wrong_drug = random.choice(all_drugs)

    # ensure it is actually wrong
    while wrong_drug in df[df["disease"] == disease]["drug"].values:
        wrong_drug = random.choice(all_drugs)

    negative_rows.append([disease, wrong_drug, 0])

negative_df = pd.DataFrame(negative_rows, columns=["disease", "drug", "label"])

# combine
final_df = pd.concat(
    [
        df[["disease", "drug", "label"]],
        negative_df,
    ]
)

# shuffle
final_df = final_df.sample(frac=1).reset_index(drop=True)

# save
final_df.to_csv("data/final_training_data.csv", index=False)

print("Training dataset ready")
print(final_df.head())
print("\nClass balance:\n", final_df["label"].value_counts())

