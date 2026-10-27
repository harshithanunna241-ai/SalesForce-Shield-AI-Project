from pathlib import Path
import pandas as pd

data_folder = Path(__file__).parent / "data"
csv_files = list(data_folder.rglob("*.csv"))

if not csv_files:
    print("No CSV files found. Check your data folder.")
else:
    print(f"Found {len(csv_files)} CSV files.")

    for file in csv_files:
        print(f"\nFile: {file.name}")
        print(f"Size: {file.stat().st_size / (1024 * 1024):.2f} MB")

        try:
            df = pd.read_csv(file, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.head())
        except Exception as e:
            print("Could not read file:", e)
