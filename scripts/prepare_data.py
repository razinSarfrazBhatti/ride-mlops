import pandas as pd

url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"

print("Downloading taxi dataset...")

df = pd.read_parquet(url)

print(f"Downloaded {len(df)} records")

df = df[
    (df["trip_distance"] > 0)
    & (df["fare_amount"] > 0)
    & (df["fare_amount"] < 200)
]

df = df.sample(n=min(10000, len(df)), random_state=42)

df.to_csv("data/taxi_sample.csv", index=False)

print(f"Saved {len(df)} trips successfully!")

print(df[["trip_distance", "fare_amount"]].head())
