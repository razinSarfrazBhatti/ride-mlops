import pandas as pd

df = pd.read_csv("data/taxi_sample.csv")

print("\n========== DATASET OVERVIEW ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df[["trip_distance", "fare_amount"]].describe())

print("\nFirst 10 records:")
print(df[["trip_distance", "fare_amount"]].head(10))
