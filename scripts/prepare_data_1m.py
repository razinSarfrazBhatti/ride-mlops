from pathlib import Path

import pandas as pd

INPUT_FILE = "data/yellow_tripdata_2025-01.parquet"
OUTPUT_FILE = "data/taxi_cleaned_1m.csv"
SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42

print("Loading original Parquet dataset...")

if not Path(INPUT_FILE).is_file():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_FILE}. Download the Parquet file first."
    )

df = pd.read_parquet(
    INPUT_FILE,
    columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "fare_amount",
    ],
)

print(f"Original records: {len(df):,}")

# Calculate trip duration in minutes
df["trip_duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

# Extract pickup hour
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

# Apply the same cleaning rules as the original project
df = df[
    (df["trip_distance"] > 0)
    & (df["trip_distance"] <= 100)
    & (df["fare_amount"] > 0)
    & (df["fare_amount"] < 200)
    & (df["trip_duration"] > 0)
    & (df["trip_duration"] <= 180)
].copy()

print(f"Valid records after cleaning: {len(df):,}")

if len(df) < SAMPLE_SIZE:
    raise ValueError(
        f"Not enough valid records: {len(df):,} available"
    )

# Handle missing passenger counts
df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)

if df["passenger_count"].isna().any():
    raise ValueError("Passenger counts are all missing; cannot fill missing values.")

# Randomly select one million valid records
df = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

# Keep only model features and target
df = df[
    [
        "trip_distance",
        "trip_duration",
        "pickup_hour",
        "passenger_count",
        "fare_amount",
    ]
]

output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\n========== DATASET RESULTS ==========")
print(f"Final records: {len(df):,}")
print(f"Output file: {OUTPUT_FILE}")
print(f"File size: {output_path.stat().st_size / (1024 ** 2):.2f} MB")
print("\nMissing values:")
print(df.isnull().sum())
print("\nStatistics:")
print(df.describe())
