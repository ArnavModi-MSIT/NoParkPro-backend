import ast
import json
import pandas as pd
import numpy as np

df = pd.read_csv("data/violations_clustered.csv")
df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")
df["date"] = pd.to_datetime(df["created_datetime"].dt.date)

daily = (
    df.groupby(["cluster_id", "date"])
    .size()
    .reset_index(name="violations")
)

daily = daily.sort_values(["cluster_id", "date"]).reset_index(drop=True)

daily["lag_1"] = daily.groupby("cluster_id")["violations"].shift(1)
daily["lag_3_avg"] = (
    daily.groupby("cluster_id")["violations"]
    .shift(1).rolling(3).mean()
    .reset_index(level=0, drop=True)
)
daily["lag_7_avg"] = (
    daily.groupby("cluster_id")["violations"]
    .shift(1).rolling(7).mean()
    .reset_index(level=0, drop=True)
)
daily["lag_14_avg"] = (
    daily.groupby("cluster_id")["violations"]
    .shift(1).rolling(14).mean()
    .reset_index(level=0, drop=True)
)

latest = (
    daily.dropna()
    .sort_values("date")
    .groupby("cluster_id")
    .tail(1)
    .copy()
)

def get_trend(row):
    recent = row["lag_3_avg"]
    baseline = row["lag_7_avg"]
    if baseline == 0:
        return "stable", 0.0
    pct_change = ((recent - baseline) / baseline) * 100
    if pct_change >= 15:
        return "worsening", round(pct_change, 1)
    elif pct_change <= -15:
        return "improving", round(pct_change, 1)
    return "stable", round(pct_change, 1)

latest[["trend", "trend_pct"]] = latest.apply(
    lambda row: pd.Series(get_trend(row)), axis=1
)

def get_weekly_pattern(cluster_id, daily_df):
    cluster_data = daily_df[daily_df["cluster_id"] == cluster_id].tail(30)
    if len(cluster_data) < 7:
        return []
    return cluster_data[["date", "violations"]].tail(14).apply(
        lambda r: {"date": str(r["date"].date()), "violations": int(r["violations"])},
        axis=1
    ).tolist()

clusters = pd.read_csv("data/clusters.csv")[
    ["cluster_id", "centroid_lat", "centroid_lon", "police_station"]
]

result = latest.merge(clusters, on="cluster_id", how="left")

print("Trend Distribution:")
print(result["trend"].value_counts())
print()
print("Top 10 Worsening Clusters:")
worsening = result[result["trend"] == "worsening"].sort_values("trend_pct", ascending=False)
print(worsening[["cluster_id", "police_station", "violations", "lag_3_avg",
                  "lag_7_avg", "trend_pct"]].head(10).to_string(index=False))
print()
print("Top 10 Improving Clusters:")
improving = result[result["trend"] == "improving"].sort_values("trend_pct")
print(improving[["cluster_id", "police_station", "violations", "lag_3_avg",
                  "lag_7_avg", "trend_pct"]].head(10).to_string(index=False))

output_cols = [
    "cluster_id", "centroid_lat", "centroid_lon", "police_station",
    "violations", "lag_1", "lag_3_avg", "lag_7_avg",
    "trend", "trend_pct", "date"
]
result[output_cols].to_csv("data/cluster_trends.csv", index=False)

summary = {
    "worsening_clusters": int((result["trend"] == "worsening").sum()),
    "stable_clusters": int((result["trend"] == "stable").sum()),
    "improving_clusters": int((result["trend"] == "improving").sum()),
    "most_worsening": worsening.iloc[0][["cluster_id", "police_station", "trend_pct"]].to_dict() if len(worsening) else None,
    "most_improving": improving.iloc[0][["cluster_id", "police_station", "trend_pct"]].to_dict() if len(improving) else None,
}

with open("data/trend_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved cluster_trends.csv ({len(result)} clusters)")
print(f"Saved trend_summary.json")
