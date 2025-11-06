# ================== test_onnx_runtime_gpu.py ==================
import onnxruntime as ort
import numpy as np

# 1️⃣ Thiết lập chạy ONNX Runtime bằng GPU (CUDA)
providers = [
    ('CUDAExecutionProvider', {'device_id': 0}),  # GPU id = 0
    'CPUExecutionProvider'
]

# 2️⃣ Load mô hình ONNX
session = ort.InferenceSession("temperature.onnx", providers=providers)
print("✅ Đã load mô hình ONNX thành công!")

# 3️⃣ Xem thông tin input/output
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print(f"📥 Input name: {input_info.name}, shape: {input_info.shape}")
print(f"📤 Output name: {output_info.name}, shape: {output_info.shape}")

# 4️⃣ Tạo dữ liệu giả (34 feature)
x_test = np.random.rand(1, 34).astype(np.float32)

# 5️⃣ Chạy dự đoán
y_pred = session.run([output_info.name], {input_info.name: x_test})

print("🎯 Dự đoán từ mô hình ONNX (GPU):", y_pred)
