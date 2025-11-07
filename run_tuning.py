# ================== run_tuning.py ==================
import warnings
warnings.filterwarnings("ignore")

import optuna
import numpy as np
from clearml import Task, Logger
from model_helper import load_data, DEVICE
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# =============== BƯỚC 1: KHỞI TẠO CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Optuna_Tuning_4Models"
)

# =============== BƯỚC 2: TẢI DỮ LIỆU ===============
X_train, y_train = load_data('train')
X_dev, y_dev = load_data('dev')

if X_train is None or X_dev is None:
    raise FileNotFoundError("❌ Không tải được dữ liệu train hoặc dev.")

print(f"✅ Dữ liệu train: {X_train.shape}, target: {y_train.shape}")
print(f"✅ Dữ liệu dev: {X_dev.shape}, target: {y_dev.shape}")

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
        model = MultiOutputRegressor(RandomForestRegressor(**params))

    # -------- XGBOOST --------
    elif model_name == "XGBoost" and XGBRegressor is not None:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_  rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "hist",
            "device": "cuda" if str(DEVICE) == "cuda" else "cpu",
            "random_state": 42,
        }
        model = MultiOutputRegressor(XGBRegressor(**params))

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
            "device": "cuda" if str(DEVICE) == "cuda" else "cpu",
            "n_jobs": -1,
            "random_state": 42,
        }
        model = MultiOutputRegressor(LGBMRegressor(**params))

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
            "task_type": "GPU" if str(DEVICE) != "cuda" else "CPU",
            "verbose": 0,
            "random_state": 42,
        }
        model = MultiOutputRegressor(CatBoostRegressor(**params))

    else:
        raise ValueError(f"❌ Model {model_name} không khả dụng hoặc chưa được import.")

    # -------- TRAIN + DEV EVALUATION --------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)

    rmse = np.sqrt(mean_squared_error(y_dev, y_pred))
    mae = mean_absolute_error(y_dev, y_pred)
    mape = mean_absolute_percentage_error(y_dev, y_pred)

    # -------- Log ClearML --------
    Logger.current_logger().report_scalar("RMSE", model_name, rmse, trial.number)
    Logger.current_logger().report_scalar("MAE", model_name, mae, trial.number)
    Logger.current_logger().report_scalar("MAPE", model_name, mape, trial.number)

    print(f"✅ Trial {trial.number} | {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}")
    return rmse  # minimize RMSE


# =============== BƯỚC 4: CHẠY OPTUNA STUDY ===============
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="4Models_Tuning_TrainDev"
    )
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    # ====== TỔNG KẾT ======
    print("\n===== 🎯 TỔNG KẾT CUỘC THI =====")
    print("Best trial params:")
    print(study.best_trial.params)
    print(f"✅ Lowest RMSE: {study.best_value:.4f}")

    Logger.current_logger().report_text(f"Best Trial Params: {study.best_trial.params}")
    Logger.current_logger().report_scalar(
        title="Best RMSE",
        series="Optuna",
        value=study.best_value,
        iteration=study.best_trial.number
    )

    print("🎉 Hoàn tất tuning & log lên ClearML!")
