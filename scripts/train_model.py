import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mlflow.set_experiment("ride-fare-prediction")

df = pd.read_csv("data/taxi_cleaned.csv")

FEATURES = [
    "trip_distance",
    "trip_duration",
    "pickup_hour",
    "passenger_count"
]

X = df[FEATURES]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

params = {
    "n_estimators": 200,
    "max_depth": 16,
    "random_state": 42
}

with mlflow.start_run(run_name="random-forest-v3"):

    model = RandomForestRegressor(
        **params,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # Track model parameters
    mlflow.log_params(params)

    # Track evaluation metrics
    mlflow.log_metrics({
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })

    # Track model artifact
    mlflow.sklearn.log_model(
        model,
        name="fare-prediction-model"
    )

    print("\n========== MLFLOW RESULTS ==========")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2: {r2:.4f}")

print("\nExperiment tracking completed!")
