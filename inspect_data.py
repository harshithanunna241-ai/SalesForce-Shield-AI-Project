
from pathlib import Path
import pandas as pd

# Search for Parquet files inside the project folder
project_folder = Path(__file__).parent
parquet_files = list(project_folder.rglob("*.parquet"))

if not parquet_files:
    print("No Parquet files found inside the project folder.")
else:
    print("Parquet files found:")
    for file in parquet_files:
        print(file)

    # Inspect the first dataset
    file_path = parquet_files[0]
    print(f"\nLoading: {file_path.name}")

    df = pd.read_parquet(file_path)

    print("\nDataset shape (rows, columns):", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())
