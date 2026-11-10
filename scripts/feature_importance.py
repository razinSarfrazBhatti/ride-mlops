import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

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

X_train = train[FEATURES]
y_train = train["fare_amount"]

X_val = validation[FEATURES]
y_val = validation["fare_amount"]

print(f"Training records: {len(train):,}")
print(f"Validation records: {len(validation):,}")

print("\nTraining Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nCalculating permutation importance...")
result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="neg_mean_absolute_error",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": result.importances_mean,
    "std": result.importances_std
}).sort_values("importance", ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.to_string(index=False))

importance.to_csv(
    "data/feature_importance.csv",
    index=False
)

print("\nSaved to data/feature_importance.csv")
