import ast
import joblib
import numpy as np
import pandas as pd

TRAFFIC_BLOCKING_TYPES = {
    "PARKING IN A MAIN ROAD",
    "DOUBLE PARKING",
    "PARKING NEAR ROAD CROSSING",
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE"
}

model = joblib.load("models/cluster_violation_predictor.pkl")
cluster_mapping = joblib.load("models/cluster_mapping.pkl")
feature_names = joblib.load("models/feature_names.pkl")


def _load_base_dataframe():
    """
    Loads and pre-processes violations_clustered_slim.csv ONCE at import time.
    The expensive parts (datetime parsing, ast.literal_eval, blocking-type flags)
    happen exactly once per process lifetime instead of on every cache-miss
    request — this is what was causing OOM kills on Render's free tier, since
    each cache-miss re-ran ast.literal_eval + 5 .apply() passes over ~290K rows.
    """
    df = pd.read_csv(
        "data/violations_clustered_slim.csv",
        usecols=["cluster_id", "created_datetime", "violation_type", "police_station", "latitude", "longitude"],
    )
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")
    df["date"] = pd.to_datetime(df["created_datetime"].dt.date)
    df["violation_type"] = df["violation_type"].apply(ast.literal_eval)

    for vtype in TRAFFIC_BLOCKING_TYPES:
        df[vtype] = df["violation_type"].apply(lambda x: int(vtype in x)).astype("int8")
    df["is_blocking"] = df["violation_type"].apply(
        lambda x: int(bool(set(x) & TRAFFIC_BLOCKING_TYPES))
    ).astype("int8")

    # downcast to shrink in-memory footprint
    df["cluster_id"] = df["cluster_id"].astype("int32")
    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")

    # violation_type lists are no longer needed after flags are computed —
    # drop the heaviest column (list-of-strings per row) to free memory
    df = df.drop(columns=["violation_type"])

    return df


# Loaded once when this module is first imported (Flask app startup),
# not on every request.
_BASE_DF = _load_base_dataframe()

# Supported windows for run_predictions_windowed(). 30-day intentionally
# excluded — verified against the real dataset that 250 of 364 clusters
# (69%) have fewer than 30 distinct days of activity in the full 150-day
# dataset (Nov 2023 - Apr 2024). A 30-day sum would be empty or near-empty
# for most clusters, misrepresenting sparse data as a quiet month rather
# than showing the real lack of history. Only "today" and "7d" are real
# and dense enough to present as comparable across all clusters.
SUPPORTED_WINDOW_DAYS = {1, 7}


def get_risk_level(pred, p75, p90):
    if pred >= p90:
        return "HIGH"
    elif pred >= p75:
        return "MEDIUM"
    return "LOW"


def calculate_impact_score(pred, max_pred, blocking_ratio, parking_main_road, double_parking, peak_hour):
    pred_score = min((pred / max_pred) * 100, 100)
    blocking_score = min(blocking_ratio * 100, 100)
    road_score = min(parking_main_road * 5, 100)
    double_score = min(double_parking * 5, 100)
    peak_hour_score = 100 if peak_hour in [7, 8, 9, 17, 18, 19] else 0
    return round(
        pred_score * 0.45 +
        blocking_score * 0.25 +
        road_score * 0.15 +
        double_score * 0.10 +
        peak_hour_score * 0.05
    )


def recommend_resources(impact):
    if impact >= 70:
        return 4, 2
    if impact >= 50:
        return 3, 1
    if impact >= 30:
        return 2, 1
    return 1, 0


def estimate_lane_blockage(parking_main_road, double_parking, lag_7_avg):
    if lag_7_avg == 0:
        return 0.0
    blocking_density = (parking_main_road * 2 + double_parking * 1.5) / lag_7_avg
    return round(min(blocking_density * 100, 60), 1)


def run_predictions():
    df = _BASE_DF.copy()

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

    daily["blocking_ratio"] = (daily["blocking_violations"] / daily["violations"]).round(3)
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    daily = daily.sort_values(["cluster_id", "date"]).reset_index(drop=True)

    daily["lag_1"] = daily.groupby("cluster_id")["violations"].shift(1)
    daily["lag_3_avg"] = (
        daily.groupby("cluster_id")["violations"].shift(1)
        .rolling(3).mean().reset_index(level=0, drop=True)
    )
    daily["lag_7_avg"] = (
        daily.groupby("cluster_id")["violations"].shift(1)
        .rolling(7).mean().reset_index(level=0, drop=True)
    )
    daily["lag_blocking_avg"] = (
        daily.groupby("cluster_id")["blocking_violations"].shift(1)
        .rolling(3).mean().reset_index(level=0, drop=True)
    )

    temporal = pd.read_csv("data/cluster_temporal.csv")
    temporal_cols = ["cluster_id", "peak_hour", "peak_day",
                     "is_night_dominant", "is_weekend_dominant", "rush_hour_ratio"]
    daily = daily.merge(temporal[temporal_cols], on="cluster_id", how="left")

    clusters = pd.read_csv("data/clusters.csv")[
        ["cluster_id", "centroid_lat", "centroid_lon",
         "blocking_ratio", "police_station"]
    ].rename(columns={"blocking_ratio": "cluster_blocking_ratio"})
    daily = daily.merge(clusters, on="cluster_id", how="left")

    daily = daily.dropna(subset=["lag_1", "lag_3_avg", "lag_7_avg", "lag_blocking_avg"])

    latest = (
        daily.sort_values("date")
        .groupby("cluster_id")
        .tail(1)
        .copy()
    )

    missing = set(latest["cluster_id"].unique()) - set(cluster_mapping.keys())
    if missing:
        latest = latest[~latest["cluster_id"].isin(missing)]

    latest["cluster_id_encoded"] = latest["cluster_id"].map(cluster_mapping)

    X = latest.copy()
    X["cluster_id"] = X["cluster_id_encoded"]
    X = X[feature_names]

    predictions = model.predict(X)

    # Cap anomalous predictions at P95 to prevent one outlier cluster dominating scores
    p95_cap = np.percentile(predictions, 95)
    predictions_capped = np.clip(predictions, 0, p95_cap)
    latest["predicted_violations"] = predictions_capped
    latest["predicted_violations_raw"] = np.round(predictions, 2)

    p75 = np.percentile(predictions_capped, 75)
    p90 = np.percentile(predictions_capped, 90)
    max_pred = predictions_capped.max()

    latest["impact_score"] = latest.apply(
        lambda row: calculate_impact_score(
            row["predicted_violations"], max_pred,
            row["blocking_ratio"], row["parking_main_road"], row["double_parking"],
            row["peak_hour"]
        ), axis=1
    )

    latest["risk_level"] = latest["predicted_violations"].apply(
        lambda p: get_risk_level(p, p75, p90)
    )

    latest[["recommended_officers", "recommended_tow_trucks"]] = latest["impact_score"].apply(
        lambda x: pd.Series(recommend_resources(x))
    )

    latest["lane_obstruction_index"] = latest.apply(
        lambda row: estimate_lane_blockage(
            row["parking_main_road"], row["double_parking"], row["lag_7_avg"]
        ), axis=1
    )

    latest["priority_rank"] = latest["impact_score"].rank(ascending=False, method="first").astype(int)

    trend_df = None
    try:
        import pandas as pd_t
        trend_df = pd_t.read_csv("data/cluster_trends.csv")[["cluster_id", "trend", "trend_pct"]]
        latest = latest.merge(trend_df, on="cluster_id", how="left")
        latest["trend"] = latest["trend"].fillna("stable")
        latest["trend_pct"] = latest["trend_pct"].fillna(0.0)
    except Exception:
        latest["trend"] = "stable"
        latest["trend_pct"] = 0.0

    def build_reasons(row):
        reasons = []
        if row["lag_7_avg"] >= p90:
            reasons.append(f"High 7-day average ({row['lag_7_avg']:.0f} violations/day)")
        elif row["lag_7_avg"] >= p75:
            reasons.append(f"Above average activity ({row['lag_7_avg']:.0f} violations/day)")
        if row["blocking_ratio"] >= 0.5:
            reasons.append(f"{round(row['blocking_ratio']*100)}% of violations directly block traffic")
        if row["parking_main_road"] > 0:
            reasons.append(f"{int(row['parking_main_road'])} main road parking violations")
        if row["double_parking"] > 0:
            reasons.append(f"{int(row['double_parking'])} double parking violations")
        if row["peak_hour"] in [7, 8, 9, 17, 18, 19]:
            reasons.append(f"Peak activity during rush hour ({row['peak_hour']}:00)")
        if row.get("trend") == "worsening":
            reasons.append(f"Trend worsening by {abs(row.get('trend_pct', 0)):.0f}% this week")
        elif row.get("trend") == "improving":
            reasons.append(f"Trend improving by {abs(row.get('trend_pct', 0)):.0f}% this week")
        if row["is_night_dominant"]:
            reasons.append("Primarily night-time violations")
        if not reasons:
            reasons.append("Consistent violation history in this cluster")
        return reasons

    latest["reasons"] = latest.apply(build_reasons, axis=1)

    return latest, p75, p90


def run_predictions_windowed(window_days=1):
    """
    Returns the same 364 clusters and the same next-day forecast as
    run_predictions(). The ONLY thing that changes with window_days is the
    displayed historical `violations` / `blocking_violations` / `blocking_ratio`
    count per cluster — recomputed as a real sum over the trailing window,
    taken directly from _BASE_DF (no fabrication, no extrapolation).

    predicted_violations, predicted_violations_raw, impact_score, risk_level,
    recommended_officers, recommended_tow_trucks, priority_rank are NOT
    recomputed per window. There is exactly one trained model and it produces
    a single next-day forecast — there is no 7-day-ahead model. Faking a
    window-dependent forecast would mean inventing numbers, so instead these
    fields are left as the real next-day forecast regardless of window, and
    a `forecast_basis` field is added so any API consumer can tell that these
    specific fields are constant across windows.

    window_days=1 -> identical to run_predictions() (single latest day's count)
    window_days=7 -> trailing 7-day sum per cluster

    window_days=30 (or any value outside SUPPORTED_WINDOW_DAYS) raises
    ValueError on purpose — see SUPPORTED_WINDOW_DAYS comment above for why
    30-day was tested against the real dataset and dropped.
    """
    if window_days not in SUPPORTED_WINDOW_DAYS:
        raise ValueError(
            f"window_days={window_days} not supported. "
            f"Only {sorted(SUPPORTED_WINDOW_DAYS)} are available — see "
            f"SUPPORTED_WINDOW_DAYS comment in predict.py for why 30-day "
            f"was excluded (insufficient real data density per cluster)."
        )

    latest, p75, p90 = run_predictions()

    if window_days == 1:
        latest["forecast_basis"] = "next_day"
        latest["days_covered_in_window"] = 1
        return latest, p75, p90

    # window_days == 7 from here on
    df = _BASE_DF
    max_date = df["date"].max()
    min_date = max_date - pd.Timedelta(days=window_days - 1)
    windowed = df[df["date"] >= min_date]

    window_agg = (
        windowed.groupby("cluster_id")
        .agg(
            violations=("cluster_id", "count"),
            blocking_violations=("is_blocking", "sum"),
            days_covered_in_window=("date", "nunique"),
        )
        .reset_index()
    )
    window_agg["blocking_ratio"] = (
        window_agg["blocking_violations"] / window_agg["violations"]
    ).round(3)

    # Drop the single-day versions of these columns before merging in the
    # windowed versions — avoids _x/_y suffix collisions on merge.
    latest = latest.drop(columns=["violations", "blocking_violations", "blocking_ratio"])
    latest = latest.merge(window_agg, on="cluster_id", how="left")

    # A cluster with zero violations in this window won't appear in
    # window_agg at all (groupby drops empty groups) — fillna(0) here means
    # "really had zero violations in this window", not "missing/unknown".
    latest["violations"] = latest["violations"].fillna(0).astype(int)
    latest["blocking_violations"] = latest["blocking_violations"].fillna(0).astype(int)
    latest["blocking_ratio"] = latest["blocking_ratio"].fillna(0.0)
    latest["days_covered_in_window"] = latest["days_covered_in_window"].fillna(0).astype(int)
    latest["forecast_basis"] = "next_day"

    return latest, p75, p90


if __name__ == "__main__":
    result, p75, p90 = run_predictions()

    output_cols = [
        "cluster_id", "centroid_lat", "centroid_lon", "police_station",
        "predicted_violations", "predicted_violations_raw", "risk_level", "impact_score", "priority_rank",
        "lane_obstruction_index", "blocking_ratio", "peak_hour", "peak_day",
        "is_night_dominant", "is_weekend_dominant",
        "recommended_officers", "recommended_tow_trucks",
        "parking_main_road", "double_parking", "blocking_violations", "violations"
    ]

    result = result[output_cols].sort_values("priority_rank")

    print(f"Risk thresholds — P75: {p75:.1f}, P90: {p90:.1f}")
    print(f"HIGH  risk clusters : {(result['risk_level'] == 'HIGH').sum()}")
    print(f"MEDIUM risk clusters: {(result['risk_level'] == 'MEDIUM').sum()}")
    print(f"LOW   risk clusters : {(result['risk_level'] == 'LOW').sum()}")
    print()
    print(result.head(10).to_string(index=False))

    result.to_csv("data/hotspot_predictions.csv", index=False)
    print(f"\nSaved {len(result)} cluster predictions to data/hotspot_predictions.csv")

    # Quick sanity check for the windowed function — run only when this
    # file is executed directly, not on import.
    print("\n--- Windowed sanity check (window_days=7) ---")
    windowed_result, _, _ = run_predictions_windowed(7)
    print(
        windowed_result[["cluster_id", "violations", "days_covered_in_window"]]
        .sort_values("violations", ascending=False)
        .head(10)
        .to_string(index=False)
    )