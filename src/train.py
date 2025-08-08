from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.data import load_housing
from src.utils import regression_metrics


def ensure_experiment(name: str) -> None:
    """
    Ensure experiment exists with artifact_location = 'mlflow-artifacts:/'
    (so Dockerized API can fetch artifacts over HTTP).
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        # Create with HTTP-served artifact store scheme
        client.create_experiment(name=name, artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(name)


def train_and_log(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test
) -> Tuple[str, Dict[str, float]]:
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        sig = infer_signature(X_train, model.predict(X_train))
        input_ex = X_train.iloc[:2]
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=sig,
            input_example=input_ex,
        )

        return run.info.run_id, metrics


def main() -> None:
    # 1) Tracking URI (optional): let env override
    tracking = os.getenv("MLFLOW_TRACKING_URI")
    if tracking:
        mlflow.set_tracking_uri(tracking)

    root = Path(__file__).resolve().parents[1]
    data_csv = root / "data" / "raw" / "california_housing.csv"

    X, y = load_housing(data_csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2) Use a NEW experiment name by default (env can override)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "housing-experiment-docker")
    ensure_experiment(exp_name)

    candidates = [
        ("linreg", LinearRegression()),
        ("dtr", DecisionTreeRegressor(random_state=42)),
    ]

    results: Dict[str, Dict[str, float]] = {}
    run_ids: Dict[str, str] = {}
    for name, model in candidates:
        run_id, metrics = train_and_log(name, model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        run_ids[name] = run_id
        print(f"{name}: {metrics}")

    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best_run_id = run_ids[best_name]
    best_rmse = results[best_name]["rmse"]
    print(f"Best: {best_name} (rmse={best_rmse:.4f}, run_id={best_run_id})")

    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    registered_name = "housing_best_model"

    mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
    print(f"Registered model: name={mv.name}, version={mv.version}")

    client.set_registered_model_alias(
        name=registered_name,
        alias="best",
        version=mv.version,
    )
    print(f"Set alias 'best' -> {registered_name} v{mv.version}")


if __name__ == "__main__":
    main()
