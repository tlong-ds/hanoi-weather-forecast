# ================== run_tuning_hourly.py ==================
# Tuning cho bài toán dự báo nhiệt độ hourly (t+1h..t+24h)
# Dữ liệu lấy từ: data_processing_hourly/
# Cấu trúc dựa trên: run_tuning_DIRECT.py
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

# Biến toàn cục để Optuna dùng trong objective()
current_X_train, current_y_train = None, None
current_X_dev, current_y_dev = None, None


# =============== BƯỚC 1: KHỞI TẠO CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast_Hourly",
    task_name="Optuna_Tuning_Hourly_MultiStep (4 Models)"
)


# =============== BƯỚC 2: TẢI DỮ LIỆU CHO MỖI HORIZON ===============
def load_data_for_horizon(h_step: int):
    """
    Tải dữ liệu đã được preprocessing cho horizon t+{h_step}h.
    - X_train, X_dev dùng chung cho mọi horizon.
    - y_train, y_dev chọn đúng 1 cột target_temp_t+{h_step}h.
    """
    global DATA_DIR

    X_train_path = os.path.join(DATA_DIR, X_TRAIN_FILE)
    y_train_path = os.path.join(DATA_DIR, Y_TRAIN_FILE)
    X_dev_path   = os.path.join(DATA_DIR, X_DEV_FILE)
    y_dev_path   = os.path.join(DATA_DIR, Y_DEV_FILE)

    target_col = f"target_temp_t+{h_step}h"

    print(f"\nLoading data for horizon: {target_col}")
    print("  From directory:", os.path.abspath(DATA_DIR))

    try:
        X_train = pd.read_csv(X_train_path, index_col=0)
        y_train_df = pd.read_csv(y_train_path, index_col=0)

        X_dev = pd.read_csv(X_dev_path, index_col=0)
        y_dev_df = pd.read_csv(y_dev_path, index_col=0)

        if target_col not in y_train_df.columns:
            raise KeyError(f"Không tìm thấy cột {target_col} trong y_train_hourly.csv")

        # Lấy 1 cột target tương ứng horizon này
        y_train = y_train_df[target_col].values.ravel()
        y_dev   = y_dev_df[target_col].values.ravel()

        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_dev:   {X_dev.shape},   y_dev:   {y_dev.shape}")

        return X_train, y_train, X_dev, y_dev

    except FileNotFoundError as e:
        print("❌ LỖI: Không tìm thấy một trong các file dữ liệu hourly.")
        print("  Hãy đảm bảo đã chạy 'preprocessing_hourly.py'.")
        print("  Lỗi gốc:", e)
        return None, None, None, None
    except Exception as e:
        print("❌ LỖI khi load dữ liệu:", e)
        return None, None, None, None


# =============== BƯỚC 3: ĐỊNH NGHĨA OBJECTIVE FUNCTION (OPTUNA) ===============
def objective(trial):
    model_name = trial.suggest_categorical(
        "model_name",
        ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
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
    elif model_name == "XGBoost" and XGBRegressor is not None:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "hist",
            "device": "cuda" if str(DEVICE) == "cuda" else "cpu",
            "random_state": 42,
        }
        model = XGBRegressor(**params)

    # -------- LIGHTGBM --------
    elif model_name == "LightGBM" and LGBMRegressor is not None:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
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
    elif model_name == "CatBoost" and CatBoostRegressor is not None:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "loss_function": "RMSE",
            "task_type": "GPU" if str(DEVICE) == "cuda" else "CPU",
            "verbose": 0,
            "random_state": 42,
        }
        model = CatBoostRegressor(**params)

    else:
        raise ValueError(f"❌ Model {model_name} không khả dụng hoặc chưa import được.")

    # -------- TRAIN + EVAL TRÊN DEV --------
    global current_X_train, current_y_train, current_X_dev, current_y_dev

    model.fit(current_X_train, current_y_train)
    y_pred = model.predict(current_X_dev)

    rmse = np.sqrt(mean_squared_error(current_y_dev, y_pred))
    mae  = mean_absolute_error(current_y_dev, y_pred)
    mape = mean_absolute_percentage_error(current_y_dev, y_pred)

    # (tuỳ chọn) log thêm vào trial để xem phân bố
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("mape", float(mape))

    return rmse  # minimize RMSE


# =============== BƯỚC 4: CHẠY OPTUNA STUDY THEO HORIZON ===============
if __name__ == "__main__":

    logger = Logger.current_logger()

    print(f"===== 🚀 BẮT ĐẦU TUNING CHO 24 HORIZONS (t+1h .. t+{N_STEPS_AHEAD}h) =====")

    for h_step in range(1, N_STEPS_AHEAD + 1):
        horizon_str = f"t+{h_step}h"
        print(f"\n{'='*80}")
        print(f"🎯 BẮT ĐẦU TUNING CHO HORIZON: {horizon_str}")
        print(f"{'='*80}")

        # 1. Load data cho horizon này
        X_train, y_train, X_dev, y_dev = load_data_for_horizon(h_step)
        if X_train is None:
            print(f"⚠️ Bỏ qua {horizon_str} vì không load được dữ liệu.")
            continue

        # 2. Gán data vào biến global cho objective()
        current_X_train, current_y_train = X_train, y_train
        current_X_dev, current_y_dev     = X_dev, y_dev

        # 3. Tạo một Study riêng cho mỗi horizon
        study = optuna.create_study(
            direction="minimize",
            study_name=f"Tuning_4Models_{horizon_str}"
        )

        # 4. Chạy optimize
        study.optimize(objective, n_trials=60, show_progress_bar=True)

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
        # RMSE per horizon
        logger.report_scalar(
            title="Best RMSE per Horizon",
            series=horizon_str,
            value=best_rmse,
            iteration=h_step
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
