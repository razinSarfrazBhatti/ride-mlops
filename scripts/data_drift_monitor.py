import json

import mlflow
import pandas as pd
from scipy.stats import ks_2samp

DATASET = "data/taxi_chronological.csv"

FEATURES = [
    "trip_distance",
    "trip_duration",
    "pickup_hour",
    "passenger_count",
]

TRAIN_START = "2025-01-01"
TRAIN_END = "2025-01-21"
VALIDATION_START = "2025-01-21"
VALIDATION_END = "2025-01-26"

OUTPUT_FILE = "data/data_drift_summary.json"


def load_chronological_data(data_path=DATASET):
    df = pd.read_csv(data_path)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    train = df[
        (df["tpep_pickup_datetime"] >= TRAIN_START)
        & (df["tpep_pickup_datetime"] < TRAIN_END)
    ].copy()

    validation = df[
        (df["tpep_pickup_datetime"] >= VALIDATION_START)
        & (df["tpep_pickup_datetime"] < VALIDATION_END)
    ].copy()

    return train, validation


def summarize_feature_drift(train_values, validation_values, feature_name):
    train_series = pd.Series(train_values).dropna()
    validation_series = pd.Series(validation_values).dropna()

    if train_series.empty or validation_series.empty:
        raise ValueError(f"No valid values available for feature: {feature_name}")

    ks_stat, p_value = ks_2samp(train_series, validation_series)

    train_mean = float(train_series.mean())
    validation_mean = float(validation_series.mean())
    train_std = float(train_series.std(ddof=0))
    validation_std = float(validation_series.std(ddof=0))

    mean_delta = float(abs(validation_mean - train_mean))
    std_delta = float(abs(validation_std - train_std))

    drift_threshold = 0.1 * max(abs(train_mean), 1.0)
    has_drift = bool(p_value < 0.05 or mean_delta > drift_threshold)

    return {
        "feature": feature_name,
        "train_mean": train_mean,
        "validation_mean": validation_mean,
        "mean_delta": mean_delta,
        "train_std": train_std,
        "validation_std": validation_std,
        "std_delta": std_delta,
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "has_drift": has_drift,
    }


def build_drift_report(train_df, validation_df, features=FEATURES):
    report = []

    for feature in features:
        row = summarize_feature_drift(
            train_df[feature],
            validation_df[feature],
            feature,
        )
        report.append(row)

    return pd.DataFrame(report).sort_values(
        ["ks_stat", "mean_delta"],
        ascending=[False, False],
    ).reset_index(drop=True)


def main():
    train_df, validation_df = load_chronological_data()

    print("Loading chronological dataset...")
    print(f"Training:   {len(train_df):,} rows")
    print(f"Validation: {len(validation_df):,} rows")

    report = build_drift_report(train_df, validation_df)

    print("\n========== DATA DRIFT REPORT ==========")
    print(
        report[
            [
                "feature",
                "train_mean",
                "validation_mean",
                "mean_delta",
                "ks_stat",
                "p_value",
                "has_drift",
            ]
        ].to_string(index=False, float_format="%.4f")
    )

    output = report.to_dict(orient="records")
    with open(OUTPUT_FILE, "w") as file:
        json.dump(output, file, indent=2)

    mlflow.set_experiment("ride-fare-prediction")

    with mlflow.start_run(run_name="data-drift-monitoring"):
        mlflow.log_params({
            "training_period": "2025-01-01 to 2025-01-20",
            "validation_period": "2025-01-21 to 2025-01-25",
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "features": ", ".join(FEATURES),
        })

        for row in output:
            for metric_name in ["ks_stat", "p_value", "mean_delta", "std_delta"]:
                mlflow.log_metric(f"{row['feature']}_{metric_name}", row[metric_name])

    print(f"\nSaved to {OUTPUT_FILE}")
    print("Data drift metrics were logged to MLflow.")

    return report


if __name__ == "__main__":
    main()
