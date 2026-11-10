import json
import pandas as pd
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET = "data/taxi_chronological.csv"

FEATURES = [
    "trip_distance",
    "trip_duration",
    "pickup_hour",
    "passenger_count"
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
test = df[
    (df["tpep_pickup_datetime"] >= "2025-01-26")
    & (df["tpep_pickup_datetime"] < "2025-02-01")
]

X_train, y_train = train[FEATURES], train["fare_amount"]
X_val, y_val = validation[FEATURES], validation["fare_amount"]
X_test, y_test = test[FEATURES], test["fare_amount"]

print(f"Training:   {len(train):,}")
print(f"Validation: {len(validation):,}")
print(f"Final test: {len(test):,}")

# Model 1: Simple linear baseline
simple_features = ["trip_distance", "trip_duration"]
simple_model = LinearRegression()
simple_model.fit(X_train[simple_features], y_train)

# Model 2: Multiple linear regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Model 3: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)

models = {
    "simple_linear": (simple_model, simple_features),
    "multiple_linear": (linear_model, FEATURES),
    "random_forest": (rf_model, FEATURES)
}


def evaluate(model, features, X, y):
    predictions = model.predict(X[features])

    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "r2": float(r2_score(y, predictions))
    }


results = {}

for name, (model, features) in models.items():
    print(f"\nEvaluating {name}...")

    results[name] = {
        "validation": evaluate(model, features, X_val, y_val),
        "test": evaluate(model, features, X_test, y_test)
    }

print("\n========== CHRONOLOGICAL RESULTS ==========")

for name, metrics in results.items():
    print(f"\n{name.upper()}")

    for period in ["validation", "test"]:
        print(f"\n{period.upper()}")
        print(f"MAE:  ${metrics[period]['mae']:.2f}")
        print(f"RMSE: ${metrics[period]['rmse']:.2f}")
        print(f"R2:   {metrics[period]['r2']:.4f}")

mlflow.set_experiment("ride-fare-prediction")

with mlflow.start_run(run_name="chronological-evaluation-1m"):
    mlflow.log_params({
        "training_period": "2025-01-01 to 2025-01-20",
        "validation_period": "2025-01-21 to 2025-01-25",
        "test_period": "2025-01-26 to 2025-01-31",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "testing_rows": len(test),
        "random_forest_n_estimators": 200,
        "random_forest_max_depth": 16
    })

    for name, metrics in results.items():
        for period, values in metrics.items():
            mlflow.log_metrics({
                f"{name}_{period}_{metric}": value
                for metric, value in values.items()
            })

with open("data/chronological_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nResults logged to MLflow.")
print("Saved to data/chronological_results.json")
