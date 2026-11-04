import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET = "data/taxi_cleaned_1m.csv"

FEATURES = [
    "trip_distance",
    "trip_duration",
    "pickup_hour",
    "passenger_count"
]

print("Loading 1-million-row dataset...")
df = pd.read_csv(DATASET)

print(f"Loaded {len(df):,} records")

X = df[FEATURES]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training records: {len(X_train):,}")
print(f"Testing records: {len(X_test):,}")

params = {
    "n_estimators": 200,
    "max_depth": 16,
    "random_state": 42
}

mlflow.set_experiment("ride-fare-prediction")

with mlflow.start_run(run_name="random-forest-1m-v1"):

    model = RandomForestRegressor(
        **params,
        n_jobs=-1
    )

    print("\nTraining Random Forest on 800,000 records...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    mlflow.log_params(params)
    mlflow.log_param("dataset_rows", len(df))
    mlflow.log_param("training_rows", len(X_train))
    mlflow.log_param("testing_rows", len(X_test))

    mlflow.log_metrics({
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })

    mlflow.sklearn.log_model(
        model,
        name="fare-prediction-model"
    )

    print("\n========== 1M MODEL RESULTS ==========")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2:   {r2:.4f}")

print("\nMLflow experiment tracking completed!")
