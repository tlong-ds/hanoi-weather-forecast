# run_tuning.py
import optuna
from clearml import Task, Logger
from model_train import load_processed_data, train_models
from model_evaluate import load_data, calculate_metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


# =============== BƯỚC 1: KHỞI TẠO CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Optuna_Tuning_3Models"
)

# =============== BƯỚC 2: TẢI DỮ LIỆU ===============
X_train, y_train = load_processed_data()
X_dev, y_dev = load_data('dev')

if X_train is None or X_dev is None:
    raise FileNotFoundError("❌ Không tải được dữ liệu. Kiểm tra thư mục processed_data.")


# =============== BƯỚC 3: ĐỊNH NGHĨA HÀM CHO OPTUNA ===============
def objective(trial):
    # --- 3.1 Chọn model ---
    model_name = trial.suggest_categorical(
        "model_name",
        ["Linear Regression", "Random Forest", "XGBoost (MultiOutput)"]
    )

    # --- 3.2 Gợi ý hyperparameters tùy theo model ---
    if model_name == "Linear Regression":
        # LinearRegression không có hyperparam phức tạp, nên chỉ cần khởi tạo đơn giản
        model_instance = LinearRegression()

    elif model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "random_state": 42,
            "n_jobs": -1
        }
        model_instance = RandomForestRegressor(**params)

    else:  # XGBoost (MultiOutput)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }
        model_instance = MultiOutputRegressor(XGBRegressor(**params))

    # --- 3.3 Huấn luyện model ---
    trained_models = train_models(
        X_train,
        y_train,
        models_to_train={model_name: model_instance}
    )
    model = trained_models[model_name]

    # --- 3.4 Dự đoán trên tập dev ---
    y_pred = model.predict(X_dev)

    # --- 3.5 Đánh giá kết quả ---
    metrics, _ = calculate_metrics(y_dev, y_pred, model_name=model_name)
    avg_rmse = metrics["Average_RMSE"]

    # --- 3.6 Ghi log lên ClearML ---
    Logger.current_logger().report_scalar(
        title="Validation RMSE",
        series=model_name,
        value=avg_rmse,
        iteration=trial.number
    )

    print(f"✅ Trial {trial.number} ({model_name}) -> RMSE trung bình = {avg_rmse:.4f}")
    return avg_rmse


# =============== BƯỚC 4: TẠO VÀ CHẠY OPTUNA STUDY ===============
study = optuna.create_study(
    direction="minimize",
    study_name="LR_RF_XGB_Tuning"
)
study.optimize(objective, n_trials=30)

# =============== BƯỚC 5: BÁO CÁO KẾT QUẢ ===============
print("\n===== 🎯 TỔNG KẾT CUỘC THI =====")
print("Best trial params:")
print(study.best_trial.params)
print(f"Lowest RMSE: {study.best_value:.4f}")

Logger.current_logger().report_text(f"Best Trial Params: {study.best_trial.params}")
Logger.current_logger().report_scalar(
    title="Best RMSE",
    series="Optuna",
    value=study.best_value,
    iteration=study.best_trial.number
)
