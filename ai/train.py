import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

df = pd.read_csv("data/cluster_features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["cluster_id"] = df["cluster_id"].astype("category")
cluster_mapping = {
    cluster: code
    for code, cluster in enumerate(df["cluster_id"].cat.categories)
}
os.makedirs("models", exist_ok=True)
joblib.dump(cluster_mapping, "models/cluster_mapping.pkl")
df["cluster_id"] = df["cluster_id"].cat.codes

exclude = ["date", "target", "police_station", "centroid_lat", "centroid_lon", "peak_day"]
features = [c for c in df.columns if c not in exclude]

X = df[features]
y = df["target"]

print("=" * 60)
print("Target Statistics")
print("=" * 60)
print(y.describe())

tscv = TimeSeriesSplit(n_splits=5)

cv_mae, cv_rmse, cv_r2 = [], [], []
baseline_mae, baseline_rmse, baseline_r2 = [], [], []

print("\n")
print("=" * 60)
print("5 Fold TimeSeries Validation")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    b_preds = X_val["lag_7_avg"].values
    b_mae = mean_absolute_error(y_val, b_preds)
    b_rmse = np.sqrt(mean_squared_error(y_val, b_preds))
    b_r2 = r2_score(y_val, b_preds)

    cv_mae.append(mae)
    cv_rmse.append(rmse)
    cv_r2.append(val_r2)
    baseline_mae.append(b_mae)
    baseline_rmse.append(b_rmse)
    baseline_r2.append(b_r2)

    print(f"\nFold {fold}")
    print(f"Train R²       : {train_r2:.3f}")
    print(f"Valid R²       : {val_r2:.3f}")
    print(f"MAE            : {mae:.3f}")
    print(f"RMSE           : {rmse:.3f}")
    print(f"Baseline MAE   : {b_mae:.3f}")
    print(f"Baseline R²    : {b_r2:.3f}")
    print(f"Best Iteration : {model.best_iteration}")

print("\n")
print("=" * 60)
print("Cross Validation Summary")
print("=" * 60)
print(f"Model    — Avg MAE: {np.mean(cv_mae):.3f}  RMSE: {np.mean(cv_rmse):.3f}  R²: {np.mean(cv_r2):.3f}  Std R²: {np.std(cv_r2):.3f}")
print(f"Baseline — Avg MAE: {np.mean(baseline_mae):.3f}  RMSE: {np.mean(baseline_rmse):.3f}  R²: {np.mean(baseline_r2):.3f}")

print("\n")
print("=" * 60)
print("Training Final Model")
print("=" * 60)

splits = list(tscv.split(X))
_, last_val_idx = splits[-1]

final_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X, y,
    eval_set=[(X.iloc[last_val_idx], y.iloc[last_val_idx])],
    verbose=False
)

joblib.dump(final_model, "models/cluster_violation_predictor.pkl")
joblib.dump(features, "models/feature_names.pkl")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": final_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n")
print("=" * 60)
print("Feature Importance")
print("=" * 60)
print(importance.head(15))

importance.to_csv("feature_importance.csv", index=False)

print("\nModel saved to models/cluster_violation_predictor.pkl")
print("Cluster mapping saved to models/cluster_mapping.pkl")
print("Feature names saved to models/feature_names.pkl")
