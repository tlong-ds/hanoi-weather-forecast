import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def train_and_evaluate():
    """
    Step 5: Model training and evaluation.
    - Load processed train/dev/test datasets.
    - Train Linear Regression, Random Forest, and XGBoost models.
    - Evaluate them on the development set.
    - Visualize predictions and summarize performance metrics.
    """

    # --- 1. Load Datasets ---
    try:
        X_train = pd.read_csv(
            'processed_data/X_train_transformed.csv',
            index_col='datetime', parse_dates=True)
        y_train = pd.read_csv(
            'processed_data/y_train.csv',
            index_col='datetime', parse_dates=True)

        X_dev = pd.read_csv(
            'processed_data/X_dev_transformed.csv',
            index_col='datetime', parse_dates=True)
        y_dev = pd.read_csv(
            'processed_data/y_dev.csv',
            index_col='datetime', parse_dates=True)

        X_test = pd.read_csv(
            'processed_data/X_test_transformed.csv',
            index_col='datetime', parse_dates=True)
        y_test = pd.read_csv(
            'processed_data/y_test.csv',
            index_col='datetime', parse_dates=True)

        print(" Tải dữ liệu thành công!")

    except FileNotFoundError as e:
        print(" Lỗi: Không tìm thấy tệp. Vui lòng kiểm tra lại đường dẫn.")
        print(e)
        return

    # --- 2. Initialize Models ---
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=4,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
    }

    # --- 3. Train and Evaluate ---
    results = {}
    for name, model in models.items():
        print(f"\n Huấn luyện mô hình: {name} ...")
        model.fit(X_train, y_train)
        print(f" Hoàn tất huấn luyện {name}!")

        # Predict on dev set
        y_dev_pred = model.predict(X_dev)

        # Calculate metrics
        mae = mean_absolute_error(y_dev, y_dev_pred)
        mse = mean_squared_error(y_dev, y_dev_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_dev, y_dev_pred)

        results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse}

        print(f"\n Hiệu suất trên tập Development ({name})")
        print(f"  MAE:   {mae:.4f}")
        print(f"  MSE:   {mse:.4f}")
        print(f"  RMSE:  {rmse:.4f}")

        # --- 4. Scatter Plot: Actual vs Predicted ---
        plt.figure(figsize=(10, 6))
        plt.scatter(y_dev, y_dev_pred, alpha=0.4, label=name)
        plt.plot([y_dev.min(), y_dev.max()],
                 [y_dev.min(), y_dev.max()],
                 'r--', lw=2, label='Đường hoàn hảo')
        plt.title(f'{name} - Thực tế vs Dự đoán (Dev Set)')
        plt.xlabel('Nhiệt độ thực tế (°C)')
        plt.ylabel('Nhiệt độ dự đoán (°C)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- 5. Line Plot: Temperature Over Time ---
        plt.figure(figsize=(15, 7))
        plt.plot(y_dev.index, y_dev['target_temp_5d'],
                 label='Thực tế', color='blue', linewidth=2)
        plt.plot(y_dev.index, y_dev_pred,
                 label=f'{name}', linestyle='--', linewidth=1.5, color='red')
        plt.title(f'{name} - Dự đoán Nhiệt độ theo Thời gian (Dev Set)')
        plt.xlabel('Ngày')
        plt.ylabel('Nhiệt độ (°C)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # --- 6. Summary of Results ---
    print("\n Tổng hợp kết quả trên tập Development:")
    result_df = pd.DataFrame(results).T
    print(result_df.round(4))


if __name__ == '__main__':
    train_and_evaluate()
