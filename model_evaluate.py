import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# --- 1. Xác định các cột mục tiêu (Toàn cục) ---
# Biến này sẽ được import bởi các tệp khác
TARGET_COLUMNS = [
    'target_temp_t+1', 
    'target_temp_t+2', 
    'target_temp_t+3', 
    'target_temp_t+4', 
    'target_temp_t+5'
]

DATA_DIR = 'processed_data'

def load_data(data_type='dev'):
    """
    Tải dữ liệu evaluation đã qua xử lý (dev hoặc test set).

    Args:
        data_type (str): 'dev' hoặc 'test'.

    Returns:
        tuple: (X_data, y_data) hoặc (None, None) nếu lỗi.
    """
    if data_type not in ['dev', 'test']:
        print(f"Lỗi: data_type phải là 'dev' hoặc 'test', không phải '{data_type}'")
        return None, None
        
    x_path = os.path.join(DATA_DIR, f'X_{data_type}_transformed.csv')
    y_path = os.path.join(DATA_DIR, f'y_{data_type}.csv')

    try:
        X_data = pd.read_csv(x_path, index_col='datetime', parse_dates=True)
        y_data = pd.read_csv(y_path, index_col='datetime', parse_dates=True)[TARGET_COLUMNS]

        print(f" Tải dữ liệu {data_type} thành công!")
        return X_data, y_data
    
    except FileNotFoundError as e:
        print(f" Lỗi: Không tìm thấy tệp {data_type}. Vui lòng kiểm tra lại đường dẫn.")
        print(e)
        return None, None
    except KeyError as e:
        print(f" Lỗi: Không tìm thấy cột mục tiêu trong tệp y_{data_type}.csv.")
        print(e)
        return None, None

def calculate_metrics(y_true, y_pred_array, model_name="Model"):
    """
    Tính toán các chỉ số MAE, RMSE cho dự đoán đa bước.
    
    Args:
        y_true (pd.DataFrame): DataFrame chứa giá trị thực tế.
        y_pred_array (np.array): Mảng NumPy chứa giá trị dự đoán (từ model.predict()).
        model_name (str): Tên của mô hình để in ra.

    Returns:
        tuple: (
            model_results (dict): Dictionary chứa tất cả các metrics.
            y_pred_df (pd.DataFrame): DataFrame của các giá trị dự đoán.
        )
    """
    # Chuyển đổi dự đoán (array) thành DataFrame
    y_pred_df = pd.DataFrame(
        y_pred_array, 
        index=y_true.index, 
        columns=TARGET_COLUMNS
    )
    
    model_results = {}
    print(f"\n--- Hiệu suất ({model_name}) ---")
    
    # Tính toán metrics cho từng horizon
    for horizon in TARGET_COLUMNS:
        mae = mean_absolute_error(y_true[horizon], y_pred_df[horizon])
        mse = mean_squared_error(y_true[horizon], y_pred_df[horizon])
        rmse = np.sqrt(mse)
        
        model_results[f'{horizon}_MAE'] = mae
        model_results[f'{horizon}_RMSE'] = rmse
        print(f"  {horizon}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # Tính toán metrics trung bình
    avg_mae = mean_absolute_error(y_true, y_pred_df)
    avg_mse = mean_squared_error(y_true, y_pred_df)
    avg_rmse = np.sqrt(avg_mse)
    
    model_results["Average_MAE"] = avg_mae
    model_results["Average_RMSE"] = avg_rmse
    print(f"  ---------------------------------")
    print(f"  Average All Horizons: MAE = {avg_mae:.4f}, RMSE = {avg_rmse:.4f}")

    return model_results, y_pred_df

def visualize_results(y_true, all_predictions):
    """
    Tạo biểu đồ (scatter và line plot) để so sánh giá trị thực tế
    với một hoặc nhiều bộ dự đoán. Tương thích với ClearML.

    Args:
        y_true (pd.DataFrame): DataFrame chứa giá trị thực tế.
        all_predictions (dict): Dictionary ('tên_model': y_pred_df).
    """
    
    # Tắt chế độ tương tác để ClearML tự động chụp ảnh biểu đồ
    plt.ioff() 
    
    print(f"\n Đang tạo biểu đồ trực quan hóa...")
    
    # Chỉ vẽ biểu đồ cho horizon đầu tiên và cuối cùng
    horizons_to_plot = [TARGET_COLUMNS[0], TARGET_COLUMNS[-1]]

    for horizon in horizons_to_plot:
        
        # --- Scatter Plot (Tất cả mô hình trên 1 biểu đồ) ---
        plt.figure(figsize=(10, 6))
        
        for name, y_pred_df in all_predictions.items():
            plt.scatter(y_true[horizon], y_pred_df[horizon], alpha=0.3, label=name)
        
        # Đường tham chiếu
        min_val = min(y_true[horizon].min(), y_true[horizon].min())
        max_val = max(y_true[horizon].max(), y_true[horizon].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Đường hoàn hảo')
        
        # Đã xoá title_suffix
        plt.title(f'Thực tế so với Dự đoán - Horizon: {horizon}')
        plt.xlabel('Nhiệt độ thực tế (°C)')
        plt.ylabel('Nhiệt độ dự đoán (°C)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show() # Xoá để ClearML/automation hoạt động

        # --- Line Plot (Tất cả mô hình trên 1 biểu đồ) ---
        plt.figure(figsize=(15, 7))
        
        # Vẽ giá trị thực tế
        plt.plot(y_true.index, y_true[horizon], label=f'Thực tế ({horizon})', color='black', linewidth=2.5)
        
        # Vẽ dự đoán của từng mô hình
        for name, y_pred_df in all_predictions.items():
            plt.plot(y_pred_df.index, y_pred_df[horizon], label=f'Dự đoán {name}', linestyle='--', linewidth=1.5, alpha=0.8)
            
        # Đã xoá title_suffix
        plt.title(f'Dự đoán nhiệt độ theo thời gian - Horizon: {horizon}')
        plt.xlabel('Ngày')
        plt.ylabel('Nhiệt độ (°C)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show() # Xoá để ClearML/automation hoạt động

def save_results_summary(results, output_filename="model_evaluation_results.csv"):
    """
    Tổng hợp kết quả từ dictionary và lưu ra file CSV.

    Args:
        results (dict): Dictionary lồng nhau, ví dụ: {'Model 1': {'MAE': 1.0, 'RMSE': 2.0}}.
        output_filename (str): Tên tệp CSV đầu ra.
    """
    print(f"\n Tổng hợp kết quả (Metrics):")
    result_df = pd.DataFrame(results).T
    
    # Sắp xếp các cột để dễ đọc hơn
    cols_ordered = []
    for h in TARGET_COLUMNS:
        cols_ordered.append(f'{h}_MAE')
        cols_ordered.append(f'{h}_RMSE')
    cols_ordered.append('Average_MAE')
    cols_ordered.append('Average_RMSE')
    
    final_cols = [c for c in cols_ordered if c in result_df.columns]
    
    print(result_df[final_cols].round(4))
    
    # Lưu kết quả ra file
    output_path = os.path.join(DATA_DIR, output_filename)
    result_df[final_cols].round(4).to_csv(output_path)
    print(f"\nĐã lưu kết quả đánh giá vào '{output_path}'")