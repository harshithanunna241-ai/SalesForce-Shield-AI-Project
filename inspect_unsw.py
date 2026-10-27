
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent / "data"
files = list(folder.rglob("*.csv"))

print("Data folder:", folder)
print("CSV files found:", len(files))

for file in files:
    print("\nFile:", file.name)
    try:
        df = pd.read_csv(file, nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head())
    except Exception as error:
        print("Read error:", error)
