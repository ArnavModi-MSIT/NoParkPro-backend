import pandas as pd
import numpy as np

df = pd.read_csv("data/violations_clustered.csv")
df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")
df["hour"] = df["created_datetime"].dt.hour
df["day_of_week"] = df["created_datetime"].dt.dayofweek
df["date"] = pd.to_datetime(df["created_datetime"].dt.date)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

TIME_WINDOWS = {
    "late_night":   list(range(0, 5)),
    "early_morning": list(range(5, 9)),
    "morning":      list(range(9, 12)),
    "afternoon":    list(range(12, 17)),
    "evening":      list(range(17, 21)),
    "night":        list(range(21, 24))
}

def assign_time_window(hour):
    for window, hours in TIME_WINDOWS.items():
        if hour in hours:
            return window
    return "unknown"

df["time_window"] = df["hour"].apply(assign_time_window)

hourly = (
    df.groupby(["cluster_id", "hour"])
    .size()
    .reset_index(name="count")
)

hourly_pivot = (
    hourly.pivot(index="cluster_id", columns="hour", values="count")
    .fillna(0)
)
hourly_pivot.columns = [f"hour_{h}" for h in hourly_pivot.columns]
hourly_pivot = hourly_pivot.reset_index()

daily = (
    df.groupby(["cluster_id", "day_of_week"])
    .size()
    .reset_index(name="count")
)

daily_pivot = (
    daily.pivot(index="cluster_id", columns="day_of_week", values="count")
    .fillna(0)
)
daily_pivot.columns = [f"day_{DAY_NAMES[d]}" for d in daily_pivot.columns]
daily_pivot = daily_pivot.reset_index()

window_counts = (
    df.groupby(["cluster_id", "time_window"])
    .size()
    .reset_index(name="count")
)

window_pivot = (
    window_counts.pivot(index="cluster_id", columns="time_window", values="count")
    .fillna(0)
    .reset_index()
)
window_pivot.columns = [
    f"window_{c}" if c != "cluster_id" else c
    for c in window_pivot.columns
]

peak_hour = (
    hourly.loc[hourly.groupby("cluster_id")["count"].idxmax()]
    [["cluster_id", "hour"]]
    .rename(columns={"hour": "peak_hour"})
)

peak_day = (
    daily.loc[daily.groupby("cluster_id")["count"].idxmax()]
    [["cluster_id", "day_of_week"]]
    .rename(columns={"day_of_week": "peak_day_num"})
)
peak_day["peak_day"] = peak_day["peak_day_num"].apply(lambda x: DAY_NAMES[x])

cluster_total = df.groupby("cluster_id").size().reset_index(name="total")

hourly_norm = hourly_pivot.copy()
hour_cols = [c for c in hourly_norm.columns if c.startswith("hour_")]
hourly_norm[hour_cols] = hourly_norm[hour_cols].div(
    cluster_total.set_index("cluster_id")["total"], axis=0
).fillna(0).round(4)

daily_norm = daily_pivot.copy()
day_cols = [c for c in daily_norm.columns if c.startswith("day_")]
daily_norm[day_cols] = daily_norm[day_cols].div(
    cluster_total.set_index("cluster_id")["total"], axis=0
).fillna(0).round(4)

temporal = (
    hourly_norm
    .merge(daily_norm, on="cluster_id")
    .merge(window_pivot, on="cluster_id")
    .merge(peak_hour, on="cluster_id")
    .merge(peak_day[["cluster_id", "peak_day"]], on="cluster_id")
)

RUSH_HOURS = [7, 8, 9, 17, 18, 19]
rush_hour_cols = [f"hour_{h}" for h in RUSH_HOURS if f"hour_{h}" in hourly_norm.columns]
temporal["rush_hour_ratio"] = hourly_norm[rush_hour_cols].sum(axis=1).round(4)

temporal["is_night_dominant"] = (
    temporal[["window_late_night", "window_night"]].sum(axis=1) >
    temporal[["window_early_morning", "window_morning", "window_afternoon", "window_evening"]].sum(axis=1)
).astype(int)

temporal["is_weekend_dominant"] = (
    temporal[["day_Sat", "day_Sun"]].sum(axis=1) >
    temporal[["day_Mon", "day_Tue", "day_Wed", "day_Thu", "day_Fri"]].sum(axis=1)
).astype(int)

temporal.to_csv("data/cluster_temporal.csv", index=False)

print(f"Temporal profiles built for {len(temporal)} clusters")
print(f"Columns: {temporal.shape[1]}")
print()
print("Night-dominant clusters:", temporal["is_night_dominant"].sum())
print("Weekend-dominant clusters:", temporal["is_weekend_dominant"].sum())
print()
print(temporal[["cluster_id", "peak_hour", "peak_day",
                "is_night_dominant", "is_weekend_dominant", "rush_hour_ratio"]].head(10))