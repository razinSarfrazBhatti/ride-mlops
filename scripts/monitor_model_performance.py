"""Monitor the chronological Random Forest on a period it never saw.

The model was trained only on trips before TRAIN_END (Jan 1-20) by
scripts/train_chronological.py. Here we score it on the most recent
window (Jan 26-31), treated as the "post-deployment" period, and compare
its MAE against the validation MAE recorded at training time.
"""

import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DATASET = "data/taxi_chronological.csv"
RESULTS_FILE = "data/chronological_results.json"

FEATURES = [
    "trip_distance",
    "trip_duration",
    "pickup_hour",
    "passenger_count",
]

TARGET = "fare_amount"

# Must match the training cut-off in train_chronological.py
TRAIN_END = "2025-01-21"

# Monitoring window, half-open: [start, end)
MONITOR_START = "2025-01-26"
MONITOR_END = "2025-02-01"

# Flag the model if MAE is more than 10% worse than its validation MAE
MAX_MAE_INCREASE = 0.10


def load_training_results():
    with open(RESULTS_FILE) as file:
        return json.load(file)


def load_monitoring_data():
    df = pd.read_csv(DATASET)
    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    )

    window = df[
        (df["tpep_pickup_datetime"] >= MONITOR_START)
        & (df["tpep_pickup_datetime"] < MONITOR_END)
    ].copy()

    if window.empty:
        raise ValueError(
            f"No trips found in [{MONITOR_START}, {MONITOR_END})"
        )

    # Leakage guard: never score the model on its own training period
    earliest = window["tpep_pickup_datetime"].min()
    if earliest < pd.Timestamp(TRAIN_END):
        raise ValueError(
            f"Monitoring window starts at {earliest}, before the "
            f"training cut-off {TRAIN_END}. This would be data leakage."
        )

    return window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        help="MLflow model URI (default: read from "
        f"{RESULTS_FILE})",
    )
    args = parser.parse_args()

    training_results = load_training_results()
    model_uri = args.model_uri or training_results.get("model_uri")

    if not model_uri:
        raise ValueError(
            f"No model_uri in {RESULTS_FILE}. Run "
            "scripts/train_chronological.py first or pass --model-uri."
        )

    baseline_mae = training_results["random_forest"]["validation"]["mae"]

    print(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    print("Loading monitoring data...")
    window_df = load_monitoring_data()
    print(f"Monitoring rows: {len(window_df):,}")

    print("Generating predictions...")
    y_actual = window_df[TARGET]
    y_predicted = model.predict(window_df[FEATURES])

    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = mean_squared_error(y_actual, y_predicted) ** 0.5
    r2 = r2_score(y_actual, y_predicted)

    mae_increase = (mae - baseline_mae) / baseline_mae
    degraded = mae_increase > MAX_MAE_INCREASE

    print("\n========== MODEL PERFORMANCE REPORT ==========")
    print(f"Window:        [{MONITOR_START}, {MONITOR_END})")
    print(f"MAE:           ${mae:.4f}")
    print(f"RMSE:          ${rmse:.4f}")
    print(f"R2:             {r2:.4f}")
    print(f"Baseline MAE:  ${baseline_mae:.4f} (validation, Jan 21-25)")
    print(f"MAE change:    {mae_increase:+.1%} "
          f"(limit +{MAX_MAE_INCREASE:.0%})")
    print(f"Status:        {'DEGRADED' if degraded else 'OK'}")

    mlflow.set_experiment("ride-fare-prediction")

    with mlflow.start_run(run_name="model-performance-monitoring"):
        mlflow.log_params({
            "model": "RandomForestRegressor",
            "model_uri": model_uri,
            "training_run_id": training_results.get("run_id"),
            "monitoring_window": f"[{MONITOR_START}, {MONITOR_END})",
            "monitoring_rows": len(window_df),
            "max_mae_increase": MAX_MAE_INCREASE,
        })

        mlflow.log_metrics({
            "monitoring_mae": mae,
            "monitoring_rmse": rmse,
            "monitoring_r2": r2,
            "baseline_mae": baseline_mae,
            "mae_increase": mae_increase,
            "degraded": int(degraded),
        })

    print("\nPerformance metrics logged to MLflow.")


if __name__ == "__main__":
    main()
