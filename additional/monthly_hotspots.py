import ast
import json
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
df["month_period"] = df["created_datetime"].dt.to_period("M")
df["month_label"] = df["created_datetime"].dt.strftime("%b %Y")

for vtype in TRAFFIC_BLOCKING_TYPES:
    df[vtype] = df["violation_type"].apply(lambda x: int(vtype in x))
df["is_blocking"] = df["violation_type"].apply(
    lambda x: int(bool(set(x) & TRAFFIC_BLOCKING_TYPES))
)

months = sorted(df["month_period"].unique())

PARTIAL_MONTHS = ["Apr 2024"]

monthly_data = {}

for month in months:
    month_df = df[df["month_period"] == month].copy()
    label = month_df["month_label"].iloc[0]

    coords = month_df[["latitude", "longitude"]].values
    eps_rad = EPS_METERS / 6371000

    db = DBSCAN(
        eps=eps_rad,
        min_samples=MIN_SAMPLES,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1
    )

    labels = db.fit_predict(np.radians(coords))
    month_df["cluster_id"] = labels

    clustered = month_df[month_df["cluster_id"] != -1].copy()

    n_clusters = len(set(labels)) - 1
    noise_pct = round((labels == -1).sum() / len(labels) * 100, 1)

    print(f"{label}: {len(month_df)} violations, {n_clusters} clusters, {noise_pct}% noise")

    cluster_stats = (
        clustered.groupby("cluster_id")
        .agg(
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
            total_violations=("cluster_id", "count"),
            blocking_violations=("is_blocking", "sum"),
            parking_main_road=("PARKING IN A MAIN ROAD", "sum"),
            double_parking=("DOUBLE PARKING", "sum"),
            police_station=("police_station", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )

    cluster_stats["blocking_ratio"] = (
        cluster_stats["blocking_violations"] / cluster_stats["total_violations"]
    ).round(3)

    max_violations = cluster_stats["total_violations"].max()

    cluster_stats["intensity"] = (
        cluster_stats["total_violations"] / max_violations
    ).round(3)

    cluster_stats["impact_score"] = cluster_stats.apply(
        lambda row: round(
            min((row["total_violations"] / max_violations) * 100, 100) * 0.45 +
            min(row["blocking_ratio"] * 100, 100) * 0.25 +
            min(row["parking_main_road"] * 5, 100) * 0.15 +
            min(row["double_parking"] * 5, 100) * 0.10
        ), axis=1
    )

    p75 = cluster_stats["total_violations"].quantile(0.75)
    p90 = cluster_stats["total_violations"].quantile(0.90)

    def risk_level(v):
        if v >= p90:
            return "HIGH"
        elif v >= p75:
            return "MEDIUM"
        return "LOW"

    cluster_stats["risk_level"] = cluster_stats["total_violations"].apply(risk_level)

    clusters_list = []
    for _, row in cluster_stats.iterrows():
        clusters_list.append({
            "cluster_id": int(row["cluster_id"]),
            "latitude": round(float(row["centroid_lat"]), 6),
            "longitude": round(float(row["centroid_lon"]), 6),
            "total_violations": int(row["total_violations"]),
            "blocking_violations": int(row["blocking_violations"]),
            "parking_main_road": int(row["parking_main_road"]),
            "double_parking": int(row["double_parking"]),
            "blocking_ratio": float(row["blocking_ratio"]),
            "intensity": float(row["intensity"]),
            "impact_score": float(row["impact_score"]),
            "risk_level": row["risk_level"],
            "police_station": row["police_station"]
        })

    monthly_data[label] = {
        "month": label,
        "is_partial": label in PARTIAL_MONTHS,
        "total_violations": int(len(month_df)),
        "total_clusters": n_clusters,
        "high_risk": int((cluster_stats["risk_level"] == "HIGH").sum()),
        "medium_risk": int((cluster_stats["risk_level"] == "MEDIUM").sum()),
        "low_risk": int((cluster_stats["risk_level"] == "LOW").sum()),
        "clusters": clusters_list
    }

output = {
    "months": list(monthly_data.keys()),
    "data": monthly_data
}

with open("data/monthly_hotspots.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved monthly_hotspots.json")
print(f"Months: {list(monthly_data.keys())}")
print(f"Total size: {round(len(json.dumps(output)) / 1024, 1)} KB")
