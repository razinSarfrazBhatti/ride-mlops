import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET = "data/taxi_chronological.csv"

ALL_FEATURES = [
    "trip_distance",
    "trip_duration",
    "pickup_hour",
    "passenger_count",
]

REDUCED_FEATURES = [
    "trip_distance",
    "trip_duration",
]

print("Loading chronological dataset...")
df = pd.read_csv(DATASET)
df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"]
)

train = df[df["tpep_pickup_datetime"] < "2025-01-21"]

validation = df[
    (df["tpep_pickup_datetime"] >= "2025-01-21")
    & (df["tpep_pickup_datetime"] < "2025-01-26")
]

print(f"Training rows:   {len(train):,}")
print(f"Validation rows: {len(validation):,}")

y_train = train["fare_amount"]
y_val = validation["fare_amount"]

feature_sets = {
    "all_features": ALL_FEATURES,
    "distance_duration_only": REDUCED_FEATURES,
}

results = {}

mlflow.set_experiment("ride-fare-prediction")

with mlflow.start_run(run_name="random-forest-feature-ablation"):

    for name, features in feature_sets.items():
        print(f"\nTraining {name}...")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(train[features], y_train)

        predictions = model.predict(validation[features])

        metrics = {
            "mae": float(mean_absolute_error(y_val, predictions)),
            "rmse": float(mean_squared_error(y_val, predictions) ** 0.5),
            "r2": float(r2_score(y_val, predictions)),
        }

        results[name] = metrics

        print(f"MAE:  ${metrics['mae']:.2f}")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"R2:   {metrics['r2']:.4f}")

        mlflow.log_metrics({
            f"{name}_{metric}": value
            for metric, value in metrics.items()
        })

        mlflow.log_param(f"{name}_features", ",".join(features))

print("\n========== FEATURE ABLATION RESULTS ==========")

for name, metrics in results.items():
    print(f"\n{name.upper()}")
    print(f"MAE:  ${metrics['mae']:.2f}")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"R2:   {metrics['r2']:.4f}")

print("\nExperiment logged to MLflow.")
