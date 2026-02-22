import pandas as pd

# load dataset (run from project root: python src/load_data.py)
df = pd.read_csv("data/medicine_data.csv")

# show basic info
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# keep only required columns
df = df[["Composition", "Uses"]]

# rename for convenience
df.columns = ["drug", "uses"]

# normalize drug text to emphasize salt names
df["drug"] = df["drug"].str.lower()
# remove dosage like (500mg), (10ml) etc
df["drug"] = df["drug"].str.replace(r"\(.*?\)", "", regex=True)
# remove numbers
df["drug"] = df["drug"].str.replace(r"\d+", "", regex=True)
# remove extra spaces
df["drug"] = df["drug"].str.replace(r"\s+", " ", regex=True).str.strip()

# preview
print("\nCleaned data preview:\n")
print(df.head())

# save cleaned file
df.to_csv("data/clean_medicine_data.csv", index=False)

print("\nCleaned dataset saved.")
