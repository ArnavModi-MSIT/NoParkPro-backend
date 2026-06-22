import ast
import pandas as pd
import numpy as np

TRAFFIC_BLOCKING_TYPES = {
    "PARKING IN A MAIN ROAD",
    "DOUBLE PARKING",
    "PARKING NEAR ROAD CROSSING",
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE"
}

df = pd.read_csv("data/violations_clustered.csv")
df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")
df["date"] = pd.to_datetime(df["created_datetime"].dt.date)
df["hour"] = df["created_datetime"].dt.hour
df["day_of_week"] = df["created_datetime"].dt.dayofweek

if "violation_type" in df.columns and isinstance(df["violation_type"].iloc[0], str):
    df["violation_type"] = df["violation_type"].apply(ast.literal_eval)

for vtype in TRAFFIC_BLOCKING_TYPES:
    df[vtype] = df["violation_type"].apply(lambda x: int(vtype in x))

df["is_blocking"] = df["violation_type"].apply(
    lambda x: int(bool(set(x) & TRAFFIC_BLOCKING_TYPES))
)

daily = (
    df.groupby(["cluster_id", "date"])
    .agg(
        violations=("cluster_id", "count"),
        blocking_violations=("is_blocking", "sum"),
        parking_main_road=("PARKING IN A MAIN ROAD", "sum"),
        double_parking=("DOUBLE PARKING", "sum"),
        parking_near_crossing=("PARKING NEAR ROAD CROSSING", "sum"),
        parking_near_signal=("PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS", "sum"),
        parking_opposite=("PARKING OPPOSITE TO ANOTHER PARKED VEHICLE", "sum"),
    )
    .reset_index()
)

daily["blocking_ratio"] = (
    daily["blocking_violations"] / daily["violations"]
).round(3)

daily["day_of_week"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

daily = daily.sort_values(["cluster_id", "date"]).reset_index(drop=True)

daily["lag_1"] = daily.groupby("cluster_id")["violations"].shift(1)
daily["lag_3_avg"] = (
    daily.groupby("cluster_id")["violations"]
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)
daily["lag_7_avg"] = (
    daily.groupby("cluster_id")["violations"]
    .shift(1)
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)
daily["lag_blocking_avg"] = (
    daily.groupby("cluster_id")["blocking_violations"]
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

temporal = pd.read_csv("data/cluster_temporal.csv")

temporal_cols = ["cluster_id", "peak_hour", "peak_day",
                 "is_night_dominant", "is_weekend_dominant", "rush_hour_ratio"]
daily = daily.merge(temporal[temporal_cols], on="cluster_id", how="left")

clusters = pd.read_csv("data/clusters.csv")[
    ["cluster_id", "centroid_lat", "centroid_lon",
     "blocking_ratio", "police_station"]
].rename(columns={
    "blocking_ratio": "cluster_blocking_ratio"
})

daily = daily.merge(clusters, on="cluster_id", how="left")

daily["target"] = daily.groupby("cluster_id")["violations"].shift(-1)

daily = daily.dropna()

daily.to_csv("data/cluster_features.csv", index=False)

print(f"Feature dataset shape: {daily.shape}")
print(f"Clusters covered     : {daily['cluster_id'].nunique()}")
print(f"Date range           : {daily['date'].min()} to {daily['date'].max()}")
print()
print("Columns:")
print(daily.columns.tolist())
print()
print(daily.head(3))