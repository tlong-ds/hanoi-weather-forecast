# train_and_eval_hourly.py
# Train + Evaluate + Plots cho dữ liệu hourly (t+1h..t+24h)
# Đọc các file *_hourly.csv trong data_processing_hourly/
# Lưu model + metrics + plots NGAY trong data_processing_hourly/

import os, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend để save file trong mọi môi trường
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1) Định nghĩa ----
DATA_DIR = "data_processing_hourly"
X_TRAIN = os.path.join(DATA_DIR, "X_train_transformed_hourly.csv")
Y_TRAIN = os.path.join(DATA_DIR, "y_train_hourly.csv")
X_DEV   = os.path.join(DATA_DIR, "X_dev_transformed_hourly.csv")
Y_DEV   = os.path.join(DATA_DIR, "y_dev_hourly.csv")

METRICS_OUT  = os.path.join(DATA_DIR, "model_evaluation_results_hourly.csv")
PLOT_METRICS = os.path.join(DATA_DIR, "metrics_bar_hourly.png")
PLOT_SCATTER = os.path.join(DATA_DIR, "pred_vs_actual_hourly.png")
PLOT_FI      = os.path.join(DATA_DIR, "feature_importance_hourly.png")

# 24 horizons: t+1h..t+24h
TARGET_COLUMNS = [f"target_temp_t+{k}h" for k in range(1, 25)]

# ---- 2) Load dữ liệu ----
X_tr = pd.read_csv(X_TRAIN, index_col=0, parse_dates=True)
y_tr = pd.read_csv(Y_TRAIN, index_col=0, parse_dates=True)[TARGET_COLUMNS]
X_dv = pd.read_csv(X_DEV,   index_col=0, parse_dates=True)
y_dv = pd.read_csv(Y_DEV,   index_col=0, parse_dates=True)[TARGET_COLUMNS]

print("Loaded shapes:",
      "X_train", X_tr.shape, "| y_train", y_tr.shape, "| X_dev", X_dv.shape, "| y_dev", y_dv.shape)

# ---- 3) Model (tham số bạn yêu cầu) ----
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    ),
    n_jobs=-1
)

# ---- 4) Train ----
print("\nTraining RandomForest (MultiOutput, hourly)…")
model.fit(X_tr, y_tr)
print("Training done.")

# ---- 5) Evaluate trên dev ----
print("\nEvaluating on dev…")
y_pred = model.predict(X_dv)

# MAE/RMSE cho từng horizon + trung bình
maes, rmses = [], []
for i, col in enumerate(TARGET_COLUMNS):
    mae = mean_absolute_error(y_dv.iloc[:, i], y_pred[:, i])
    rmse = mean_squared_error(y_dv.iloc[:, i], y_pred[:, i])
    maes.append(float(mae)); rmses.append(float(rmse))

metrics = {f"{TARGET_COLUMNS[i]}_MAE": maes[i] for i in range(24)}
metrics.update({f"{TARGET_COLUMNS[i]}_RMSE": rmses[i] for i in range(24)})
metrics["Average_MAE"]  = float(np.mean(maes))
metrics["Average_RMSE"] = float(np.mean(rmses))
print(f"Dev  Avg MAE: {metrics['Average_MAE']:.4f} | Avg RMSE: {metrics['Average_RMSE']:.4f}")

# ---- 6) Lưu model + metrics ----
os.makedirs(DATA_DIR, exist_ok=True)
ordered_cols = []
for c in TARGET_COLUMNS:
    ordered_cols += [f"{c}_MAE", f"{c}_RMSE"]
ordered_cols += ["Average_MAE", "Average_RMSE"]
pd.DataFrame([{k: metrics[k] for k in ordered_cols}]).to_csv(METRICS_OUT, index=False)

print("\nSaved:")
print("  Metrics->", os.path.abspath(METRICS_OUT))

# ---- 7) Plots ----
# 7.1. Bar plot MAE/RMSE theo horizon
horizons = np.arange(1, 25)
width = 0.4
plt.figure(figsize=(12, 5))
plt.bar(horizons - width/2, maes, width, label="MAE")
plt.bar(horizons + width/2, rmses, width, label="RMSE")
plt.xticks(horizons, [f"+{h}h" for h in horizons], rotation=0)
plt.xlabel("Horizon"); plt.ylabel("Error")
plt.title("Hourly Forecast Errors per Horizon (Dev)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_METRICS, dpi=150)
plt.close()
print("  Plot  ->", os.path.abspath(PLOT_METRICS))

# 7.2. Scatter Actual vs Predicted cho t+1h và t+24h
def scatter_ap(y_true_col, y_pred_col, label, ax):
    ax.scatter(y_true_col, y_pred_col, s=8, alpha=0.5)
    # đường y=x
    lo = min(y_true_col.min(), y_pred_col.min())
    hi = max(y_true_col.max(), y_pred_col.max())
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(label)

plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
scatter_ap(y_dv.iloc[:, 0].values, y_pred[:, 0], "t+1h", ax1)
ax2 = plt.subplot(1, 2, 2)
scatter_ap(y_dv.iloc[:, -1].values, y_pred[:, -1], "t+24h", ax2)
plt.tight_layout()
plt.savefig(PLOT_SCATTER, dpi=150)
plt.close()
print("  Plot  ->", os.path.abspath(PLOT_SCATTER))

# 7.3. Feature importance (trung bình trên 24 RF con) – Top-20
try:
    # Lấy importances từ từng estimator và trung bình
    importances = np.stack([est.feature_importances_ for est in model.estimators_], axis=0)
    mean_importance = importances.mean(axis=0)
    fi = pd.Series(mean_importance, index=X_tr.columns).sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    fi.iloc[::-1].plot(kind="barh")  # vẽ từ nhỏ đến lớn để đẹp
    plt.xlabel("Mean Feature Importance")
    plt.title("Top-20 Feature Importances (avg over horizons)")
    plt.tight_layout()
    plt.savefig(PLOT_FI, dpi=150)
    plt.close()
    print("  Plot  ->", os.path.abspath(PLOT_FI))
except Exception as e:
    print("  (Skip feature importance plot):", e)

print("\nAll done.")
