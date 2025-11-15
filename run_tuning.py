# ================== run_tuning_DIRECT.py ==================
# <<< THAY ĐỔI >>> Phiên bản này được sửa để chạy 5 quy trình tuning riêng biệt
# cho 5 bộ dữ liệu đã được feature selection riêng
# ========================================================

import warnings
warnings.filterwarnings("ignore")

import optuna
import numpy as np
import pandas as pd # <<< THAY ĐỔI >>> Cần thêm pandas để load data
import os # <<< THAY ĐỔI >>> Cần thêm os
from clearml import Task, Logger
from model_helper import DEVICE # Giả sử DEVICE vẫn được định nghĩa trong model_helper
from sklearn.ensemble import RandomForestRegressor
# <<< THAY ĐỔI >>> KHÔNG CẦN MultiOutputRegressor nữa
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# =============== BƯỚC 1: KHỞI TẠO CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Optuna_Tuning_Direct_Strategy (5 Models)" # <<< THAY ĐỔI >>> Tên task mới
)

# <<< THAY ĐỔI >>> Biến toàn cục để lưu trữ data cho mỗi trial
# Chúng sẽ được cập nhật bên trong vòng lặp ở main
current_X_train, current_y_train = None, None
current_X_dev, current_y_dev = None, None

# <<< THAY ĐỔI >>> Định nghĩa số ngày dự báo
N_STEPS_AHEAD = 5

# =============== BƯỚC 2: TẢI DỮ LIỆU (ĐỊNH NGHĨA HÀM MỚI) ===============
def load_data_for_day(day_step):
    """
    Tải bộ dữ liệu đã được xử lý riêng cho ngày t+{day_step}
    """
    day_str = f"t_{day_step}"
    data_dir = f'processed_data/target_{day_str}'
    
    print(f"\nLoading data from: {data_dir}")
    
    try:
        X_train = pd.read_csv(os.path.join(data_dir, f'X_train_{day_str}.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(data_dir, f'y_train_{day_str}.csv'), index_col=0)
        
        X_dev = pd.read_csv(os.path.join(data_dir, f'X_dev_{day_str}.csv'), index_col=0)
        y_dev = pd.read_csv(os.path.join(data_dir, f'y_dev_{day_str}.csv'), index_col=0)
        
        # <<< THAY ĐỔI >>> Quan trọng: y_train/y_dev giờ chỉ là 1 cột.
        # Cần .values.ravel() để biến nó thành mảng 1D mà model mong đợi
        return X_train, y_train.values.ravel(), X_dev, y_dev.values.ravel()
    
    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy file dữ liệu cho {day_str}. Bạn đã chạy preprocessing.py chưa?")
        print(e)
        return None, None, None, None


# =============== BƯỚC 3: ĐỊNH NGHĨA OBJECTIVE FUNCTION (OPTUNA) ===============
# <<< THAY ĐỔI >>> Hàm objective giờ sẽ dùng data toàn cục
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
        # <<< THAY ĐỔI >>> Không còn MultiOutputRegressor
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
        # <<< THAY ĐỔI >>> Không còn MultiOutputRegressor
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
            "device": "gpu",
            "device_type": "cuda",
            "random_state": 42,
        }
        # <<< THAY ĐỔI >>> Không còn MultiOutputRegressor
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
        # <<< THAY ĐỔI >>> Không còn MultiOutputRegressor
        model = CatBoostRegressor(**params)

    else:
        raise ValueError(f"❌ Model {model_name} không khả dụng hoặc chưa được import.")

    # -------- TRAIN + DEV EVALUATION --------
    # <<< THAY ĐỔI >>> Dùng biến toàn cục đã được set trong vòng lặp main
    global current_X_train, current_y_train, current_X_dev, current_y_dev
    
    model.fit(current_X_train, current_y_train)
    y_pred = model.predict(current_X_dev)

    # <<< THAY ĐỔI >>> y_dev giờ là mảng 1D, nên việc tính toán vẫn giữ nguyên
    rmse = np.sqrt(mean_squared_error(current_y_dev, y_pred))
    mae = mean_absolute_error(current_y_dev, y_pred)
    mape = mean_absolute_percentage_error(current_y_dev, y_pred)

    # -------- Log ClearML --------
    # Không log scalar ở đây nữa, sẽ log tổng kết ở main
    
    # print(f"Trial {trial.number}: {model_name}, RMSE={rmse:.4f}")
    return rmse  # minimize RMSE


# =============== BƯỚC 4: CHẠY OPTUNA STUDY (TRONG VÒNG LẶP) ===============
if __name__ == "__main__":
    
    # <<< THAY ĐỔI >>> Tạo một logger ClearML
    logger = Logger.current_logger()
    
    print(f"===== 🚀 BẮT ĐẦU 5 QUY TRÌNH TUNING (CHO t+1 ĐẾN t+{N_STEPS_AHEAD}) =====")
    
    # <<< THAY ĐỔI >>> Vòng lặp chính, chạy 5 lần
    for day_step in range(1, N_STEPS_AHEAD + 1):
        day_str = f"t+{day_step}"
        print(f"\n{'='*70}")
        print(f"🎯 BẮT ĐẦU TUNING CHO NGÀY: {day_str}")
        print(f"{'='*70}")
        
        # 1. Tải data cho ngày này
        X_train, y_train, X_dev, y_dev = load_data_for_day(day_step)
        if X_train is None:
            continue # Bỏ qua nếu không tải được
            
        # 2. Gán data vào biến toàn cục để hàm objective thấy
        current_X_train, current_y_train = X_train, y_train
        current_X_dev, current_y_dev = X_dev, y_dev

        print(f"✅ Dữ liệu {day_str} train: {X_train.shape}, target: {y_train.shape}")
        print(f"✅ Dữ liệu {day_str} dev: {X_dev.shape}, target: {y_dev.shape}")
        
        # 3. Tạo một Study MỚI cho ngày này
        study = optuna.create_study(
            direction="minimize",
            study_name=f"Tuning_4Models_{day_str}"
        )
        
        # 4. Chạy optimize
        study.optimize(objective, n_trials=60, show_progress_bar=True)

        # 5. Lấy kết quả tốt nhất cho ngày này
        best_params = study.best_trial.params
        best_rmse = study.best_value
        best_model_name = best_params.get("model_name", "N/A")

        print(f"\n===== 🎯 TỔNG KẾT CHO {day_str} =====")
        print(f"  Best Model: {best_model_name}")
        print(f"  Best RMSE: {best_rmse:.4f}")
        print(f"  Best Params: {best_params}")

        # 6. Log kết quả tốt nhất cho ngày này lên ClearML
        logger.report_scalar(
            title="Best RMSE per Day",
            series=f"{day_str}",
            value=best_rmse,
            iteration=day_step
        )
        logger.report_scalar(
            title="Best Model per Day",
            series=f"{day_str}",
            value=best_model_name, # Log tên model
            iteration=day_step
        )
        logger.report_text(f"Best Params {day_str}: {best_params}")

    print("\n🎉🎉🎉 Hoàn tất CẢ 5 quy trình tuning & log lên ClearML! 🎉🎉🎉")