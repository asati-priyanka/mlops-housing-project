from pathlib import Path
import pandas as pd
from typing import Tuple

def load_housing(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    # California Housing (sklearn) target column:
    target_col = "MedHouseVal"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
