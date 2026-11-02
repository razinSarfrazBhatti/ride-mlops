import pandas as pd

INPUT_FILE = "data/taxi_sample.csv"
OUTPUT_FILE = "data/taxi_cleaned.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

initial_count = len(df)

# Convert timestamps
df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"], errors="coerce"
)

df["tpep_dropoff_datetime"] = pd.to_datetime(
    df["tpep_dropoff_datetime"], errors="coerce"
)

# Calculate trip duration in minutes
df["trip_duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

# Extract pickup hour
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

# Remove invalid records and extreme outliers
df = df[
    (df["trip_distance"] > 0)
    & (df["trip_distance"] <= 100)
    & (df["fare_amount"] > 0)
    & (df["fare_amount"] < 200)
    & (df["trip_duration"] > 0)
    & (df["trip_duration"] <= 180)
]

# Handle missing passenger counts
df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)

# Keep only the features needed for our first model
df = df[
    [
        "trip_distance",
        "trip_duration",
        "pickup_hour",
        "passenger_count",
        "fare_amount"
    ]
]

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== CLEANING RESULTS ==========")
print(f"Original records: {initial_count}")
print(f"Clean records: {len(df)}")
print(f"Removed records: {initial_count - len(df)}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nCleaned data statistics:")
print(df.describe())

print(f"\nSaved cleaned dataset to {OUTPUT_FILE}")
