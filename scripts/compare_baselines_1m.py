import pandas as pd
import mlflow
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET = "data/taxi_cleaned_1m.csv"

FEATURES = [
    "trip_distance",
    "trip_duration",
    "pickup_hour",
    "passenger_count"
]

print("Loading dataset...")
if not Path(DATASET).is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}. Run scripts/prepare_data_1m.py first."
    )

df = pd.read_csv(DATASET)

required_columns = [*FEATURES, "fare_amount"]
missing_columns = sorted(set(required_columns) - set(df.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

df = df[required_columns].dropna()
if df.empty:
    raise ValueError("Dataset has no complete rows to train on.")

X = df[FEATURES]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training records: {len(X_train):,}")
print(f"Testing records:  {len(X_test):,}")

# Model 1: Simple linear formula (distance + duration)
simple_features = ["trip_distance", "trip_duration"]

simple_model = LinearRegression()
simple_model.fit(X_train[simple_features], y_train)
simple_predictions = simple_model.predict(X_test[simple_features])

# Model 2: Multiple linear regression (all four features)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# Model 3: Random Forest (all four features)
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=16,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


def evaluate(name, actual, predicted):
    return {
        "model": name,
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "r2": r2_score(actual, predicted)
    }


results = [
    evaluate("Simple Linear Baseline", y_test, simple_predictions),
    evaluate("Multiple Linear Regression", y_test, linear_predictions),
    evaluate("Random Forest", y_test, rf_predictions)
]

print("\n========== THREE-MODEL COMPARISON ==========")

for result in results:
    print(f"\n{result['model']}")
    print(f"MAE:  ${result['mae']:.2f}")
    print(f"RMSE: ${result['rmse']:.2f}")
    print(f"R2:   {result['r2']:.4f}")

mlflow.set_experiment("ride-fare-prediction")

with mlflow.start_run(run_name="three-model-comparison-1m"):
    mlflow.log_params({
        "dataset_rows": len(df),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "split_random_state": 42,
        "random_forest_n_estimators": 200,
        "random_forest_max_depth": 16
    })

    for result, prefix in zip(
        results,
        ["simple_linear", "multiple_linear", "random_forest"]
    ):
        mlflow.log_metrics({
            f"{prefix}_mae": result["mae"],
            f"{prefix}_rmse": result["rmse"],
            f"{prefix}_r2": result["r2"]
        })

print("\nThree-model comparison logged to MLflow.")
