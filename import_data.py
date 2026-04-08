"""
Download the UCI Early Stage Diabetes Risk Prediction Dataset directly.
"""
import urllib.request
import zipfile
import os
from pathlib import Path

DATA_DIR = Path(r"d:\Miniproject\data")
DATA_DIR.mkdir(exist_ok=True)

url = "https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip"
zip_path = DATA_DIR / "diabetes_symptoms.zip"

print("Downloading dataset from UCI...")
urllib.request.urlretrieve(url, zip_path)
print(f"Downloaded: {zip_path}")

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(DATA_DIR)
    print(f"Extracted files: {z.namelist()}")

os.remove(zip_path)
print("Done!")

# List what we got
for f in DATA_DIR.iterdir():
    print(f"  {f.name} ({f.stat().st_size} bytes)")
