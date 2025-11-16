# ================== run_tuning_hourly.py ==================
# Tuning cho bài toán dự báo nhiệt độ hourly (t+1h..t+24h)
# Dữ liệu lấy từ: data_processing_hourly/
# Cấu trúc dựa trên: run_tuning_DIRECT.py
#
# PHIÊN BẢN ĐÃ SỬA:
# 1. Chỉ load data 1 lần (ngoài vòng lặp)
# 2. Bỏ biến global, dùng lambda
# 3. Thêm `log=True` cho learning_rate
# 4. Cải thiện ClearML logging
# 5. Kiểm tra model khả dụng
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import optuna

from clearml import Task, Logger

# Nếu bạn có sẵn model_helper.DEVICE thì giữ dòng sau,
# còn không thì comment lại và dùng DEVICE = "cpu"
try:
    from model_helper import DEVICE
except ImportError:
    DEVICE = "cpu"

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

# =============== CONSTANTS ===============
DATA_DIR = "data_processing_hourly"
X_TRAIN_FILE = "X_train_transformed_hourly.csv"
Y_TRAIN_FILE = "y_train_hourly.csv"
X_DEV_FILE   = "X_dev_transformed_hourly.csv"
Y_DEV_FILE   = "y_dev_hourly.csv"

# 24 horizons: t+1h..t+24h
N_STEPS_AHEAD = 24


# =============== BƯỚC 1: KHỞI TẠO CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast_Hourly",
    task_name="Optuna_Tuning_Hourly_MultiStep (4 Models)"
)

# =============== BƯỚC 2: TẢI DỮ LIỆU (ĐÃ BỎ) ===============
# Hàm load_data_for_horizon đã được tích hợp vào main()
# để tránh đọc file 24 lần.


# =============== BƯỚC 3: ĐỊNH NGHĨA OBJECTIVE FUNCTION (OPTUNA) ===============
# Thêm tham số (X_train, y_train, X_dev, y_dev) để tránh dùng global
def objective(trial, X_train, y_train, X_dev, y_dev):
    
    # [FIX 2] Tự động build danh sách model đã cài
    available_models = ["Random Forest"]
    if XGBRegressor is not None:
        available_models.append("XGBoost")
    if LGBMRegressor is not None:
        available_models.append("LightGBM")
    if CatBoostRegressor is not None:
        available_models.append("CatBoost")

    model_name = trial.suggest_categorical(
        "model_name",
        available_models # Chỉ chọn từ các model khả dụng
    )

    # -------- RANDOM FOREST --------
    if model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)

    # -------- XGBOOST --------
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), # [FIX 5]
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "hist",
            "device": "cuda" if str(DEVICE) == "cuda" else "cpu",
            "random_state": 42,
        }
        model = XGBRegressor(**params)

    # -------- LIGHTGBM --------
    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), # [FIX 5]
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "device_type": "cpu",
            "n_jobs": -1,
            "random_state": 42,
        }
        model = LGBMRegressor(**params)

    # -------- CATBOOST --------
    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True), # [FIX 5]
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "loss_function": "RMSE",
            "task_type": "GPU" if str(DEVICE) == "cuda" else "CPU",
            "verbose": 0,
            "random_state": 42,
        }
        model = CatBoostRegressor(**params)

    # -------- TRAIN + EVAL TRÊN DEV --------
    # [FIX 3] Không cần global, dùng tham số truyền vào
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)

    rmse = np.sqrt(mean_squared_error(y_dev, y_pred))
    mae  = mean_absolute_error(y_dev, y_pred)
    mape = mean_absolute_percentage_error(y_dev, y_pred)

    # (tuỳ chọn) log thêm vào trial để xem phân bố
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("mape", float(mape))

    return rmse  # minimize RMSE


# =============== BƯỚC 4: CHẠY OPTUNA STUDY THEO HORIZON ===============
if __name__ == "__main__":

    logger = Logger.current_logger()

    # [FIX 1] Tải dữ liệu 1 LẦN DUY NHẤT
    print("🚀 Đang tải dữ liệu X, y (DataFrame)...")
    try:
        X_train_path = os.path.join(DATA_DIR, X_TRAIN_FILE)
        y_train_path = os.path.join(DATA_DIR, Y_TRAIN_FILE)
        X_dev_path   = os.path.join(DATA_DIR, X_DEV_FILE)
        y_dev_path   = os.path.join(DATA_DIR, Y_DEV_FILE)

        X_train = pd.read_csv(X_train_path, index_col=0)
        y_train_df = pd.read_csv(y_train_path, index_col=0)
        X_dev = pd.read_csv(X_dev_path, index_col=0)
        y_dev_df = pd.read_csv(y_dev_path, index_col=0)
        
        print(f"  Tải thành công X_train: {X_train.shape}, y_train_df: {y_train_df.shape}")
        print(f"  Tải thành công X_dev:   {X_dev.shape},   y_dev_df:   {y_dev_df.shape}")

    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không thể tải dữ liệu ban đầu. Dừng chương trình.")
        print(f"  Kiểm tra lại đường dẫn: {os.path.abspath(DATA_DIR)}")
        print(f"  Lỗi gốc: {e}")
        exit() # Thoát nếu không load được file

    print(f"===== 🚀 BẮT ĐẦU TUNING CHO 24 HORIZONS (t+1h .. t+{N_STEPS_AHEAD}h) =====")

    for h_step in range(1, N_STEPS_AHEAD + 1):
        horizon_str = f"t+{h_step}h"
        target_col = f"target_temp_t+{h_step}h" # Đảm bảo tên cột này chính xác

        print(f"\n{'='*80}")
        print(f"🎯 BẮT ĐẦU TUNING CHO HORIZON: {horizon_str} (Cột: {target_col})")
        print(f"{'='*80}")

        # 1. [FIX 1] Lấy dữ liệu y cho horizon này (không đọc file)
        if target_col not in y_train_df.columns:
            print(f"⚠️ Cảnh báo: Không tìm thấy cột {target_col} trong y_train_df. Bỏ qua horizon này.")
            continue
        
        y_train = y_train_df[target_col].values.ravel()
        y_dev   = y_dev_df[target_col].values.ravel()

        # 2. [FIX 3] Không cần gán vào biến global

        # 3. Tạo một Study riêng cho mỗi horizon
        study = optuna.create_study(
            direction="minimize",
            study_name=f"Tuning_4Models_{horizon_str}"
        )

        # 4. [FIX 3] Chạy optimize dùng lambda để truyền data
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_dev, y_dev), 
            n_trials=60, 
            show_progress_bar=True
        )

        # 5. Lấy kết quả tốt nhất
        best_params = study.best_trial.params
        best_rmse   = study.best_value
        best_model_name = best_params.get("model_name", "N/A")

        print(f"\n===== 🎯 TỔNG KẾT CHO {horizon_str} =====")
        print(f"  Best Model: {best_model_name}")
        print(f"  Best RMSE:  {best_rmse:.4f}")
        print(f"  Best Params:")
        for k, v in best_params.items():
            print(f"    - {k}: {v}")

        # 6. Log kết quả lên ClearML
        
        # [FIX 4] Log RMSE vào 1 biểu đồ duy nhất
        logger.report_scalar(
            title="Best RMSE per Horizon",  # Tên biểu đồ
            series="RMSE",                  # Tên đường line
            value=best_rmse,                # Giá trị (trục Y)
            iteration=h_step                # Horizon (trục X)
        )

        # Log model name
        logger.report_text(
            f"[{horizon_str}] Best Model: {best_model_name}",
            level="INFO"
        )

        # Log full params
        logger.report_text(
            f"[{horizon_str}] Best Params: {best_params}"
        )

    print("\n🎉🎉🎉 Hoàn tất tuning cho toàn bộ 24 horizons & log lên ClearML! 🎉🎉🎉")