import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import os # Thêm import os để lưu file

# --- 1. Xác định các cột mục tiêu (Toàn cục) ---
TARGET_COLUMNS = [
    'target_temp_t+1', 
    'target_temp_t+2', 
    'target_temp_t+3', 
    'target_temp_t+4', 
    'target_temp_t+5'
]

def load_processed_data():
    """
    Tải dữ liệu đã qua xử lý (train set).
    """
    try:
        X_train = pd.read_csv('processed_data/X_train_transformed.csv', index_col='datetime', parse_dates=True)
        y_train = pd.read_csv('processed_data/y_train.csv', index_col='datetime', parse_dates=True)[TARGET_COLUMNS]

        print(" Tải dữ liệu thành công!")
        return X_train, y_train
    
    except FileNotFoundError as e:
        print(f" Lỗi: Không tìm thấy tệp. Vui lòng kiểm tra lại đường dẫn.")
        print(e)
        return None, None
    except KeyError as e:
        print(f" Lỗi: Không tìm thấy cột mục tiêu trong tệp y_...csv.")
        print(f" Vui lòng kiểm tra lại danh sách 'TARGET_COLUMNS' trong code.")
        print(e)
        return None, None

def train_models(X_train, y_train, models_to_train=None):
    """
    Khởi tạo và huấn luyện các mô hình trên dữ liệu training.
    Trả về một dictionary chứa các mô hình đã được huấn luyện.
    """
    # --- Khởi tạo mô hình ---
    if models_to_train is None:
        models_to_train = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_split=4, 
                random_state=42, n_jobs=-1
            ),
            "XGBoost (MultiOutput)": MultiOutputRegressor(XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ))
        }
    
    trained_models = {}

    for name, model in models_to_train.items():
        print(f"\n Huấn luyện mô hình: {name} ...")
        model.fit(X_train, y_train)
        print(f" Hoàn tất huấn luyện {name}!")
        trained_models[name] = model
        
    return trained_models

