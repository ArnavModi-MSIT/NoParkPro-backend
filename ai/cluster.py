import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

TRAFFIC_BLOCKING_TYPES = {
    "PARKING IN A MAIN ROAD",
    "DOUBLE PARKING",
    "PARKING NEAR ROAD CROSSING",
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE"
}

EPS_METERS = 150
MIN_SAMPLES = 10

df = pd.read_csv("data/violations.csv")
df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")
df["violation_type"] = df["violation_type"].apply(ast.literal_eval)
df["hour"] = df["created_datetime"].dt.hour
df["day_of_week"] = df["created_datetime"].dt.dayofweek
df["date"] = df["created_datetime"].dt.date

coords = df[["latitude", "longitude"]].values
eps_rad = EPS_METERS / 6371000

db = DBSCAN(
    eps=eps_rad,
    min_samples=MIN_SAMPLES,
    algorithm="ball_tree",
    metric="haversine",
    n_jobs=-1
)

print("Running DBSCAN on", len(coords), "records...")
labels = db.fit_predict(np.radians(coords))

df["cluster_id"] = labels

noise = (labels == -1).sum()
n_clusters = len(set(labels)) - 1
print(f"Clusters found : {n_clusters}")
print(f"Noise points   : {noise} ({noise/len(labels)*100:.1f}%)")

df_clustered = df[df["cluster_id"] != -1].copy()

for vtype in TRAFFIC_BLOCKING_TYPES:
    df_clustered[vtype] = df_clustered["violation_type"].apply(lambda x: int(vtype in x))

df_clustered["is_blocking"] = df_clustered["violation_type"].apply(
    lambda x: int(bool(set(x) & TRAFFIC_BLOCKING_TYPES))
)

cluster_stats = (
    df_clustered.groupby("cluster_id")
    .agg(
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        total_violations=("cluster_id", "count"),
        blocking_violations=("is_blocking", "sum"),
        unique_days=("date", "nunique"),
        peak_hour=("hour", lambda x: x.value_counts().idxmax()),
        peak_day=("day_of_week", lambda x: x.value_counts().idxmax()),
        police_station=("police_station", lambda x: x.value_counts().idxmax()),
    )
    .reset_index()
)

for vtype in TRAFFIC_BLOCKING_TYPES:
    col_stats = (
        df_clustered.groupby("cluster_id")[vtype]
        .sum()
        .reset_index()
        .rename(columns={vtype: vtype.replace(" ", "_").lower()})
    )
    cluster_stats = cluster_stats.merge(col_stats, on="cluster_id")

cluster_stats["blocking_ratio"] = (
    cluster_stats["blocking_violations"] / cluster_stats["total_violations"]
).round(3)

cluster_stats["violations_per_day"] = (
    cluster_stats["total_violations"] / cluster_stats["unique_days"]
).round(2)

cluster_stats = cluster_stats.sort_values("total_violations", ascending=False).reset_index(drop=True)

print(f"\nTop 10 clusters by total violations:")
print(cluster_stats[["cluster_id", "centroid_lat", "centroid_lon", "total_violations",
                       "blocking_ratio", "violations_per_day", "peak_hour", "police_station"]].head(10))

cluster_stats.to_csv("data/clusters.csv", index=False)
df_clustered.to_csv("data/violations_clustered.csv", index=False)
joblib.dump(db, "models/dbscan_model.pkl")

print(f"\nSaved {n_clusters} clusters to data/clusters.csv")
print(f"Saved labelled violations to data/violations_clustered.csv")
print(f"Saved DBSCAN model to models/dbscan_model.pkl")
