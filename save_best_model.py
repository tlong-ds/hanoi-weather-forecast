# ================== save_best_model.py ==================
from catboost import CatBoostRegressor
import joblib
import numpy as np
from model_train import load_processed_data
from model_evaluate import load_data

# 1️⃣ Load dữ liệu train + dev
X_train, y_train = load_processed_data()
X_dev, y_dev = load_data('dev')

# 2️⃣ Gộp dữ liệu và chỉ lấy cột đầu tiên của y (temperature)
X_full = np.vstack([X_train, X_dev])
y_full = np.vstack([y_train, y_dev])[:, 0]  # chỉ lấy cột đầu tiên

print("✅ X_full shape:", X_full.shape)
print("✅ y_full shape:", y_full.shape)

# 3️⃣ Dùng tham số tốt nhất từ Optuna
best_params = {
    'iterations': 300,
    'depth': 9,
    'learning_rate': 0.014211216422573586,
    'l2_leaf_reg': 4.335449095664048,
    'subsample': 0.7365113132247191,
    'bootstrap_type': 'Bernoulli',
    'loss_function': 'RMSE',
    'task_type': 'GPU',              # Dùng GPU
    'devices': '0',                  # GPU id = 0
    'random_seed': 42,
    'verbose': 100
}

# 4️⃣ Huấn luyện mô hình
model = CatBoostRegressor(**best_params)
model.fit(X_full, y_full)

# 5️⃣ Lưu mô hình lại
joblib.dump(model, "best_model.pkl")
print("✅ Đã huấn luyện xong mô hình tốt nhất và lưu vào 'best_model.pkl'")
