from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def download_data(save_path: str = "data/raw/housing.csv") -> None:
    """
    Downloads the California Housing dataset and saves it as a CSV file.
    
    Args:
        save_path (str): Path to save the downloaded CSV file.
    """
    print("📥 Downloading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Data saved to: {save_path}")

if __name__ == "__main__":
    download_data()
