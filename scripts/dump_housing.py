from pathlib import Path
from sklearn.datasets import fetch_california_housing

root = Path(__file__).resolve().parents[1]  # project root
out = root / "data" / "raw" / "california_housing.csv"

df = fetch_california_housing(as_frame=True).frame
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved: {out}")
