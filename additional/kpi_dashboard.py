import ast
import json
import pandas as pd
import numpy as np

df = pd.read_csv("data/violations.csv")
df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")
df["violation_type"] = df["violation_type"].apply(ast.literal_eval)
df["date"] = df["created_datetime"].dt.date
df["month_label"] = df["created_datetime"].dt.strftime("%b %Y")
df["hour"] = df["created_datetime"].dt.hour
df["day_of_week"] = df["created_datetime"].dt.day_name()

all_violation_types = []
for vt in df["violation_type"]:
    all_violation_types.extend(vt)
violation_type_series = pd.Series(all_violation_types)

daily_counts = df.groupby("date").size().reset_index(name="count")
daily_counts["date"] = pd.to_datetime(daily_counts["date"])
daily_counts = daily_counts.sort_values("date")

monthly_counts = (
    df.groupby("month_label")
    .size()
    .reset_index(name="violations")
)
month_order = ["Nov 2023", "Dec 2023", "Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024"]
monthly_counts["month_order"] = monthly_counts["month_label"].map(
    {m: i for i, m in enumerate(month_order)}
)
monthly_counts = monthly_counts.sort_values("month_order").drop(columns="month_order")

approved = (df["validation_status"] == "approved").sum()
rejected = (df["validation_status"] == "rejected").sum()
total_validated = approved + rejected
approval_rate = round((approved / total_validated) * 100, 1) if total_validated > 0 else 0

scita_sent = (df["data_sent_to_scita"] == True).sum()
scita_rate = round((scita_sent / len(df)) * 100, 1)

top_junctions = (
    df[df["junction_name"] != "No Junction"]
    .groupby("junction_name")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="violations")
)

top_zones = (
    df.groupby("police_station")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="violations")
)

vehicle_type_dist = (
    df["vehicle_type"]
    .value_counts()
    .head(8)
    .reset_index()
    .rename(columns={"vehicle_type": "vehicle_type", "count": "count"})
)

violation_type_dist = (
    violation_type_series
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={0: "violation_type", "count": "count"})
)

hourly_dist = (
    df.groupby("hour")
    .size()
    .reset_index(name="violations")
)

day_dist = (
    df.groupby("day_of_week")
    .size()
    .reset_index(name="violations")
)
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_dist["order"] = day_dist["day_of_week"].map({d: i for i, d in enumerate(day_order)})
day_dist = day_dist.sort_values("order").drop(columns="order")

validation_dist = (
    df["validation_status"]
    .value_counts()
    .reset_index()
    .rename(columns={"validation_status": "status", "count": "count"})
)

zone_approval = (
    df[df["validation_status"].isin(["approved", "rejected"])]
    .groupby("police_station")
    .apply(lambda x: round((x["validation_status"] == "approved").sum() / len(x) * 100, 1))
    .reset_index(name="approval_rate")
    .sort_values("approval_rate", ascending=False)
)

kpi = {
    "overview": {
        "total_violations": int(len(df)),
        "unique_vehicles": int(df["vehicle_number"].nunique()),
        "total_zones": int(df["police_station"].nunique()),
        "total_junctions": int((df["junction_name"] != "No Junction").sum()),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
            "days": int((pd.to_datetime(df["date"].max()) - pd.to_datetime(df["date"].min())).days)
        },
        "avg_violations_per_day": round(len(df) / daily_counts["date"].nunique(), 1),
        "peak_day_violations": int(daily_counts["count"].max()),
        "peak_day_date": str(daily_counts.loc[daily_counts["count"].idxmax(), "date"].date())
    },
    "enforcement": {
        "total_approved": int(approved),
        "total_rejected": int(rejected),
        "approval_rate_pct": approval_rate,
        "scita_sent_pct": scita_rate,
        "validation_breakdown": validation_dist.to_dict(orient="records")
    },
    "top_junctions": top_junctions.to_dict(orient="records"),
    "top_zones": top_zones.to_dict(orient="records"),
    "vehicle_type_distribution": vehicle_type_dist.to_dict(orient="records"),
    "violation_type_distribution": violation_type_dist.to_dict(orient="records"),
    "hourly_distribution": hourly_dist.to_dict(orient="records"),
    "day_of_week_distribution": day_dist.to_dict(orient="records"),
    "monthly_trend": monthly_counts.to_dict(orient="records"),
    "zone_approval_rates": zone_approval.head(10).to_dict(orient="records")
}

with open("data/kpi_dashboard.json", "w") as f:
    json.dump(kpi, f, indent=2)

print("=" * 50)
print("KPI Summary")
print("=" * 50)
print(f"Total violations     : {kpi['overview']['total_violations']:,}")
print(f"Unique vehicles      : {kpi['overview']['unique_vehicles']:,}")
print(f"Avg violations/day   : {kpi['overview']['avg_violations_per_day']}")
print(f"Peak day             : {kpi['overview']['peak_day_date']} ({kpi['overview']['peak_day_violations']} violations)")
print(f"Approval rate        : {approval_rate}%")
print(f"SCITA sent rate      : {scita_rate}%")
print()
print("Top 5 Junctions:")
print(top_junctions.head(5).to_string(index=False))
print()
print("Top 5 Vehicle Types:")
print(vehicle_type_dist.head(5).to_string(index=False))
print()
print("Top 5 Violation Types:")
print(violation_type_dist.head(5).to_string(index=False))
print()
print(f"Saved kpi_dashboard.json ({round(len(json.dumps(kpi))/1024, 1)} KB)")
