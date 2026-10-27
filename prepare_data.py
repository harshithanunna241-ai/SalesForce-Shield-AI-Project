
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

train_path = DATA_DIR / "UNSW_NB15_testing-set.csv"
test_path = DATA_DIR / "UNSW_NB15_training-set.csv"

# Load the prepared training and testing datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

# Check that both datasets contain the binary target label
if "label" not in train_df.columns or "label" not in test_df.columns:
    raise ValueError("The 'label' column is missing from a dataset.")

# Remove rows with missing target labels
train_df = train_df.dropna(subset=["label"]).copy()
test_df = test_df.dropna(subset=["label"]).copy()

# Display class distribution
print("\nTraining label counts:")
print(train_df["label"].value_counts())

print("\nTesting label counts:")
print(test_df["label"].value_counts())

# Save cleaned copies without changing the original files
output_dir = DATA_DIR / "prepared"
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "training_clean.csv", index=False)
test_df.to_csv(output_dir / "testing_clean.csv", index=False)

print("\nPrepared datasets saved in:", output_dir)
