"""
MLOps pipeline using Prefect.
Install: pip install prefect scikit-learn mlflow pandas
Run:     python pipeline_example.py
"""
import json
import logging
import time
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ── Tasks ─────────────────────────────────────────────────────────
@task(name="load-data", retries=2, retry_delay_seconds=5)
def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    logger = get_run_logger()
    logger.info("Loading breast cancer dataset")
    ds = load_breast_cancer()
    X, y = ds.data.astype(np.float32), ds.target
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.mean():.2%} positive")
    return X, y, list(ds.feature_names)


@task(name="validate-data")
def validate_data(X: np.ndarray, y: np.ndarray) -> bool:
    logger = get_run_logger()
    checks = {
        "no_nan": not np.isnan(X).any(),
        "no_inf": not np.isinf(X).any(),
        "positive_samples": X.shape[0] > 0,
        "class_balance": 0.1 < y.mean() < 0.9,
    }
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {check}")
    all_passed = all(checks.values())
    if not all_passed:
        raise ValueError(f"Data validation failed: {[k for k, v in checks.items() if not v]}")
    return True


@task(name="preprocess-data")
def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


@task(name="train-model")
def train_model(X_train, y_train, model_type: str = "random_forest", **hyperparams):
    logger = get_run_logger()
    logger.info(f"Training {model_type} with {hyperparams}")

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            random_state=42, n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 3),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    logger.info(f"Training completed in {train_time:.2f}s")
    return model, train_time


@task(name="evaluate-model")
def evaluate_model(model, X_test, y_test) -> dict:
    logger = get_run_logger()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
    }
    for name, val in metrics.items():
        logger.info(f"  {name}: {val:.4f}")
    return metrics


@task(name="validate-model-performance")
def validate_performance(metrics: dict, thresholds: dict) -> bool:
    logger = get_run_logger()
    passed = True
    for metric, threshold in thresholds.items():
        actual = metrics.get(metric, 0)
        ok = actual >= threshold
        logger.info(f"  [{('PASS' if ok else 'FAIL')}] {metric}: {actual:.4f} >= {threshold}")
        if not ok:
            passed = False
    return passed


@task(name="log-to-mlflow")
def log_to_mlflow(model, metrics: dict, hyperparams: dict, scaler, train_time: float,
                  model_type: str, run_name: str = "pipeline-run"):
    mlflow.set_experiment("ml-pipeline")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**hyperparams, "model_type": model_type})
        mlflow.log_metrics({**metrics, "training_time_s": train_time})
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id
    return run_id


@task(name="save-model")
def save_model(model, scaler, output_dir: str = "models"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{output_dir}/model.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    return output_dir


# ── Main Flow ─────────────────────────────────────────────────────
@flow(name="ml-training-pipeline", log_prints=True)
def training_pipeline(
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = None,
    min_accuracy: float = 0.90,
    min_f1: float = 0.88,
):
    logger = get_run_logger()
    logger.info(f"Starting pipeline: {model_type}")

    # Data stage
    X, y, feature_names = load_data()
    validate_data(X, y)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Training stage
    hyperparams = {"n_estimators": n_estimators, "max_depth": max_depth}
    model, train_time = train_model(X_train, y_train, model_type=model_type, **hyperparams)

    # Evaluation stage
    metrics = evaluate_model(model, X_test, y_test)
    thresholds = {"accuracy": min_accuracy, "f1": min_f1, "auc_roc": 0.85}
    passed = validate_performance(metrics, thresholds)

    if not passed:
        logger.error("Model did not meet performance thresholds!")
        raise ValueError("Model validation failed — not saving.")

    # Artifact stage
    run_id = log_to_mlflow(model, metrics, hyperparams, scaler, train_time, model_type)
    model_dir = save_model(model, scaler)

    logger.info(f"Pipeline complete. MLflow run: {run_id}, Model saved: {model_dir}")
    return {"metrics": metrics, "run_id": run_id, "model_dir": model_dir}


if __name__ == "__main__":
    result = training_pipeline(
        model_type="random_forest",
        n_estimators=100,
        min_accuracy=0.90,
    )
    print(f"\nResult: {json.dumps(result['metrics'], indent=2)}")
