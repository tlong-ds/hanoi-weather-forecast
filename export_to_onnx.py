# ================== export_to_onnx.py ==================
from catboost import CatBoostRegressor
import joblib

# 1️⃣ Load lại mô hình CatBoost đã huấn luyện
model: CatBoostRegressor = joblib.load("best_model.pkl")

# 2️⃣ Xuất mô hình sang ONNX
model.save_model("temperature.onnx", format="onnx")

print("✅ Mô hình đã được export sang ONNX: temperature.onnx")
