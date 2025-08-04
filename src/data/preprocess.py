import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    input_path: str = "data/raw/housing.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Loads raw data, preprocesses it (scaling), and saves train/test sets.
    
    Args:
        input_path (str): Path to raw data CSV.
        output_dir (str): Directory to save preprocessed data.
        test_size (float): Proportion of test set.
        random_state (int): Random seed.
    """
    print("⚙️ Preprocessing data...")
    df = pd.read_csv(input_path)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to DataFrame again
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    X_train_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"✅ Preprocessed data saved to {output_dir}/")

if __name__ == "__main__":
    preprocess_data()
