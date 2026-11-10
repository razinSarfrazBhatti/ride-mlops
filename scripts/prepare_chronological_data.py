import pandas as pd

INPUT_FILE = "data/yellow_tripdata_2025-01.parquet"
OUTPUT_FILE = "data/taxi_chronological.csv"

print("Loading original Parquet dataset...")

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

# Keep January records only
df = df[
    (df["tpep_pickup_datetime"] >= "2025-01-01")
    & (df["tpep_pickup_datetime"] < "2025-02-01")
].copy()

# Apply existing cleaning rules
df = df[
    (df["trip_distance"] > 0)
    & (df["trip_distance"] <= 100)
    & (df["fare_amount"] > 0)
    & (df["fare_amount"] < 200)
    & (df["trip_duration"] > 0)
    & (df["trip_duration"] <= 180)
].copy()

# Handle missing passenger counts
df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)

# Preserve pickup timestamp for chronological splitting
df = df[
    [
        "tpep_pickup_datetime",
        "trip_distance",
        "trip_duration",
        "pickup_hour",
        "passenger_count",
        "fare_amount",
    ]
]

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== CHRONOLOGICAL DATASET ==========")
print(f"Total valid records: {len(df):,}")
print(f"Earliest pickup: {df['tpep_pickup_datetime'].min()}")
print(f"Latest pickup: {df['tpep_pickup_datetime'].max()}")
print(f"Output: {OUTPUT_FILE}")
print(f"File size: {pd.io.common.file_exists(OUTPUT_FILE)}")
print("\nMissing values:")
print(df.isnull().sum())
