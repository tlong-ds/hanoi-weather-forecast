# ================== run_tuning.py (FAST PARALLEL VERSION) ==================
import warnings
warnings.filterwarnings("ignore")

import optuna
import numpy as np
from clearml import Task, Logger
from model_train import load_processed_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import multiprocessing


# ====== Optional imports ======
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


# =============== BƯỚC 1: KHỞI TẠO CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Optuna_Tuning_5Models_GPU_PARALLEL"
)

# =============== BƯỚC 2: TẢI DỮ LIỆU ===============
X_train, y_train = load_processed_data()
if X_train is None:
    raise FileNotFoundError("❌ Không tải được dữ liệu train.")

print(f"✅ Dữ liệu train: {X_train.shape}, target: {y_train.shape}")


# =============== BƯỚC 3: CROSS-VALIDATION EVALUATION ===============
def evaluate_model_cv(model, X, y, n_splits=3):
    """Đánh giá trung bình RMSE, MAE, R², MAPE qua K-Fold"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmses, maes, r2s, mapes = [], [], [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        maes.append(mean_absolute_error(y_val, y_pred))
        r2s.append(r2_score(y_val, y_pred))
        mapes.append(mean_absolute_percentage_error(y_val, y_pred))

    return np.mean(rmses), np.mean(maes), np.mean(r2s), np.mean(mapes)


# =============== GPU CHECK HELPERS ===============
def _check_gpu_xgb():
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(tree_method="gpu_hist", device="cuda")
        model.fit([[0, 0]], [0])
        return True
    except Exception:
        return False


def _check_gpu_lgbm():
    try:
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(device="gpu")
        m.fit([[0, 0]], [0])
        return True
    except Exception:
        return False


def _check_gpu_catboost():
    try:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(task_type="GPU", iterations=5, verbose=0)
        m.fit([[0, 0]], [0])
        return True
    except Exception:
        return False


# =============== BƯỚC 4: OBJECTIVE FUNCTION (OPTUNA) ===============
def objective(trial):
    model_name = trial.suggest_categorical(
        "model_name",
        ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    )

    # -------- Linear Regression --------
    if model_name == "Linear Regression":
        model = MultiOutputRegressor(LinearRegression())

    # -------- Random Forest --------
    elif model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = MultiOutputRegressor(RandomForestRegressor(**params))

    # -------- XGBoost --------
    elif model_name == "XGBoost" and XGBRegressor is not None:
        gpu_ok = _check_gpu_xgb()
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "gpu_hist" if gpu_ok else "hist",
            "device": "cuda" if gpu_ok else "cpu",
            "random_state": 42,
        }
        model = MultiOutputRegressor(XGBRegressor(**params))

    # -------- LightGBM --------
    elif model_name == "LightGBM" and LGBMRegressor is not None:
        gpu_ok = _check_gpu_lgbm()
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "device": "gpu" if gpu_ok else "cpu",
            "n_jobs": -1,
            "random_state": 42,
        }
        model = MultiOutputRegressor(LGBMRegressor(**params))

    # -------- CatBoost --------
    elif model_name == "CatBoost" and CatBoostRegressor is not None:
        gpu_ok = _check_gpu_catboost()
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "loss_function": "RMSE",
            "task_type": "GPU" if gpu_ok else "CPU",
            "verbose": 0,
            "random_state": 42,
        }
        model = MultiOutputRegressor(CatBoostRegressor(**params))
    else:
        raise ValueError(f"❌ Model {model_name} không khả dụng.")

    # -------- Đánh giá model --------
    avg_rmse, avg_mae, avg_r2, avg_mape = evaluate_model_cv(model, X_train, y_train, n_splits=3)

    # -------- Log ClearML --------
    Logger.current_logger().report_scalar("RMSE", model_name, avg_rmse, trial.number)
    Logger.current_logger().report_scalar("MAE", model_name, avg_mae, trial.number)
    Logger.current_logger().report_scalar("R2", model_name, avg_r2, trial.number)
    Logger.current_logger().report_scalar("MAPE", model_name, avg_mape, trial.number)

    print(f"✅ Trial {trial.number} | {model_name}: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}, MAPE={avg_mape:.4f}")
    return avg_rmse  # mục tiêu: minimize RMSE


# =============== BƯỚC 5: TẠO VÀ CHẠY OPTUNA SONG SONG ===============
if __name__ == "__main__":
    n_jobs = max(1, multiprocessing.cpu_count() - 1)  # tận dụng tối đa CPU

    study = optuna.create_study(
        direction="minimize",
        study_name="5Models_Tuning_GPU_Parallel",
    )

    study.optimize(objective, n_trials=60, n_jobs=n_jobs, show_progress_bar=True)

    # ====== TỔNG KẾT ======
    print("\n===== 🎯 TỔNG KẾT =====")
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
