
from pathlib import Path
import pandas as pd

# Locate the training dataset
project_folder = Path(__file__).parent
data_folder = project_folder / "data"
file_path = data_folder / "UNSW_NB15_training-set.csv"

print("Loading training dataset...")

df = pd.read_csv(file_path)

print("\n--- Dataset Information ---")
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Target Label Counts ---")
if "label" in df.columns:
    print(df["label"].value_counts(dropna=False))
else:
    print("No 'label' column found.")

print("\n--- Missing Values ---")
print("Total missing values:", int(df.isnull().sum().sum()))

if "attack_cat" in df.columns:
    print("\n--- Attack Categories ---")
    print(df["attack_cat"].value_counts(dropna=False))
