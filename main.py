
from pathlib import Path
import pandas as pd

# Locate the dataset folder
DATASET_DIR = Path(__file__).parent / "dataset"

# Find all Parquet files
parquet_files = list(DATASET_DIR.glob("*.parquet"))

print("Number of Parquet files found:", len(parquet_files))

if not parquet_files:
    print("No Parquet files found in the dataset folder.")
else:
    # Inspect the first file before loading everything
    file_path = parquet_files[0]
    print("\nInspecting file:", file_path.name)

    df = pd.read_parquet(file_path)

    print("\nDataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)
