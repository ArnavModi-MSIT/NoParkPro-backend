import ast
import json
import pandas as pd

df = pd.read_csv("data/violations.csv")
df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")
df["violation_type"] = df["violation_type"].apply(ast.literal_eval)
df["date"] = df["created_datetime"].dt.date
df["month_label"] = df["created_datetime"].dt.strftime("%b %Y")

df["primary_violation"] = df["violation_type"].apply(lambda x: x[0] if x else "UNKNOWN")

vehicle_stats = (
    df.groupby("vehicle_number")
    .agg(
        total_violations=("vehicle_number", "count"),
        first_violation=("date", "min"),
        last_violation=("date", "max"),
        unique_zones=("police_station", "nunique"),
        top_zone=("police_station", lambda x: x.value_counts().idxmax()),
        vehicle_type=("vehicle_type", lambda x: x.value_counts().idxmax()),
    )
    .reset_index()
)

violation_type_counts = (
    df.groupby("vehicle_number")["primary_violation"]
    .value_counts()
    .reset_index()
    .rename(columns={"primary_violation": "violation_type", "count": "count"})
)

top_violation = (
    violation_type_counts
    .sort_values("count", ascending=False)
    .groupby("vehicle_number")
    .first()
    .reset_index()[["vehicle_number", "violation_type"]]
    .rename(columns={"violation_type": "top_violation_type"})
)

vehicle_stats = vehicle_stats.merge(top_violation, on="vehicle_number", how="left")

vehicle_stats["span_days"] = (
    pd.to_datetime(vehicle_stats["last_violation"]) -
    pd.to_datetime(vehicle_stats["first_violation"])
).dt.days + 1

vehicle_stats["violations_per_month"] = (
    vehicle_stats["total_violations"] / (vehicle_stats["span_days"] / 30)
).round(2)

repeat_offenders = vehicle_stats[vehicle_stats["total_violations"] >= 5].copy()
repeat_offenders = repeat_offenders.sort_values("total_violations", ascending=False).reset_index(drop=True)
repeat_offenders["rank"] = repeat_offenders.index + 1

repeat_offenders.to_csv("data/repeat_offenders.csv", index=False)

print(f"Total repeat offenders (5+ violations): {len(repeat_offenders)}")
print(f"10+ violations: {(repeat_offenders['total_violations'] >= 10).sum()}")
print(f"20+ violations: {(repeat_offenders['total_violations'] >= 20).sum()}")
print()
print(repeat_offenders[[
    "rank", "vehicle_number", "total_violations", "top_zone",
    "top_violation_type", "vehicle_type", "unique_zones"
]].head(15).to_string(index=False))

zone_offender_counts = (
    repeat_offenders.groupby("top_zone")
    .agg(
        repeat_offender_count=("vehicle_number", "count"),
        total_repeat_violations=("total_violations", "sum"),
        top_vehicle=("vehicle_number", "first")
    )
    .reset_index()
    .sort_values("repeat_offender_count", ascending=False)
    .reset_index(drop=True)
)
zone_offender_counts.to_csv("data/zone_repeat_offenders.csv", index=False)

summary = {
    "total_unique_vehicles": int(df["vehicle_number"].nunique()),
    "repeat_offenders_5plus": int(len(repeat_offenders)),
    "repeat_offenders_10plus": int((repeat_offenders["total_violations"] >= 10).sum()),
    "repeat_offenders_20plus": int((repeat_offenders["total_violations"] >= 20).sum()),
    "top_offender_vehicle": repeat_offenders.iloc[0]["vehicle_number"],
    "top_offender_violations": int(repeat_offenders.iloc[0]["total_violations"]),
    "top_offender_zone": repeat_offenders.iloc[0]["top_zone"],
    "most_common_violation": repeat_offenders["top_violation_type"].value_counts().idxmax(),
    "top_zones": zone_offender_counts.head(10).to_dict(orient="records")
}

with open("data/repeat_offenders_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved repeat_offenders.csv ({len(repeat_offenders)} records)")
print(f"Saved zone_repeat_offenders.csv ({len(zone_offender_counts)} zones)")
print(f"Saved repeat_offenders_summary.json")
