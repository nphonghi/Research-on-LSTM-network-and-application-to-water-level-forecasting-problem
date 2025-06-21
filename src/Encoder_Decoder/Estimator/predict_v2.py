import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error as sklearn_mae
from src.Encoder_Decoder.PreprocessData.prepare_data import prepare_data 
from src.Encoder_Decoder.Model.model_v2 import EncoderDecoderModel, Encoder, Decoder 

# --- Cấu hình ---
new_data_file = 'data/DataScaled/Test_WL_2024.csv'
model_load_path = 'models/encoder_decoder_lstm_model_v2.keras'

attributes = ['WL_KienGiang', 'WL_DongHoi', 'WL_LeThuy', 'RF_DongHoi', 'Tide_DongHoi', 'RF_KienGiang', 'RF_LeThuy']
target_attribute = 'WL_LeThuy' 
T = 24  # Độ dài cửa sổ quá khứ (lookback)
P = 12  # Độ dài chuỗi dự báo (horizon)
n_input_features = len(attributes)
n_output_features = 1 # Chỉ dự báo mực nước

# --- Load mô hình ---
print(f"Đang tải mô hình từ: {model_load_path}")
# Khi load mô hình tùy chỉnh, cần cung cấp các lớp tùy chỉnh đó
custom_objects = {
    'EncoderDecoderModel': EncoderDecoderModel,
    'Encoder': Encoder,
    'Decoder': Decoder
}
try:
    loaded_model = tf.keras.models.load_model(model_load_path, custom_objects=custom_objects)
    print("Mô hình đã được tải thành công.")
    loaded_model.summary() # In summary để kiểm tra
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

# --- Chuẩn bị dữ liệu mới ---
print(f"\nĐang chuẩn bị dữ liệu từ: {new_data_file}")
# Hàm prepare_data sẽ trả về encoder_inputs và decoder_targets
# decoder_targets chính là y_true để so sánh
encoder_inputs_new, y_true_new_data = prepare_data(
    new_data_file, attributes, T, P, target_attribute
)

# Chuyển đổi sang tensor
encoder_inputs_new_tf = tf.convert_to_tensor(encoder_inputs_new, dtype=tf.float32)
y_true_new_data_tf = tf.convert_to_tensor(y_true_new_data, dtype=tf.float32)

# Tạo input dictionary cho mô hình (chỉ cần encoder_inputs khi predict)
new_input_dict = {
    'encoder_inputs': encoder_inputs_new_tf
}
print(f"Số lượng mẫu trong dữ liệu mới: {encoder_inputs_new_tf.shape[0]}")

# --- Thực hiện dự đoán ---
print("\nBắt đầu dự đoán trên dữ liệu mới...")
try:
    y_pred_new_np = loaded_model.predict(new_input_dict)
    # y_pred_new_np = loaded_model(new_input_dict, training=False).numpy() # Cách gọi call trực tiếp
    print(f"Dự đoán hoàn tất. Shape của kết quả dự đoán: {y_pred_new_np.shape}")
except Exception as e:
    print(f"Lỗi trong quá trình dự đoán: {e}")
    exit()

# --- Đánh giá kết quả ---
y_true_new_data_np = y_true_new_data_tf.numpy()

scaler = joblib.load('data/DataScaled/minmax_scaler2.pkl')

shape = y_pred_new_np.shape  
y_pred_2d = y_pred_new_np.reshape(-1, shape[-1])  # [batch_size * timesteps, n_features]
y_pred_2d_original = scaler.inverse_transform(y_pred_2d)
y_pred_new_np = y_pred_2d_original.reshape(shape)

shape = y_true_new_data_np.shape  
y_true_2d = y_true_new_data_np.reshape(-1, shape[-1])  # [batch_size * timesteps, n_features]
y_true_2d_original = scaler.inverse_transform(y_true_2d)
y_true_new_data_np = y_true_2d_original.reshape(shape)

if y_true_new_data_np is not None and len(y_true_new_data_np) > 0:
    print("\nĐánh giá mô hình trên dữ liệu mới:")

    y_true_flat = y_true_new_data_np.reshape(-1)
    y_pred_flat = y_pred_new_np.reshape(-1)

    mse_overall = mean_squared_error(y_true_flat, y_pred_flat)
    rmse_overall = np.sqrt(mse_overall)
    mae_overall = sklearn_mae(y_true_flat, y_pred_flat)
    r2_overall = r2_score(y_true_flat, y_pred_flat)

    print(f"Overall MSE: {mse_overall:.4f}")
    print(f"Overall RMSE: {rmse_overall:.4f}")
    print(f"Overall MAE: {mae_overall:.4f}")
    print(f"Overall R2 Score: {r2_overall:.4f}")

    print("\nĐánh giá tại từng bước dự báo (horizon) trên dữ liệu mới:")
    for step in range(P):
        y_true_step = y_true_new_data_np[:, step, 0]
        y_pred_step = y_pred_new_np[:, step, 0]
        
        mse_step = mean_squared_error(y_true_step, y_pred_step)
        rmse_step = np.sqrt(mse_step)
        mae_step = sklearn_mae(y_true_step, y_pred_step)
        r2_step = r2_score(y_true_step, y_pred_step)
        
        print(f"  Horizon t+{step+1}: RMSE={rmse_step:.4f}, MAE={mae_step:.4f}, R2={r2_step:.4f}")
else:
    print("\nKhông có dữ liệu thực tế (y_true) để đánh giá trên tập dữ liệu mới.")


# --- Vẽ biểu đồ cho một vài mẫu ---
num_samples_to_plot = min(3, encoder_inputs_new_tf.shape[0]) # Vẽ tối đa 3 mẫu
if y_true_new_data_np is not None:
    for i in range(num_samples_to_plot):
        plt.figure(figsize=(12, 6))
        plt.plot(range(P), y_true_new_data_np[i, :, 0], label=f'Thực tế (Mẫu {i})', marker='o')
        plt.plot(range(P), y_pred_new_np[i, :, 0], label=f'Dự đoán (Mẫu {i})', linestyle='--', marker='x')
        plt.title(f'Dự báo mực nước cho {P} bước tới trên dữ liệu mới (Mẫu {i})')
        plt.xlabel(f'Bước thời gian dự báo (t+1 đến t+{P})')
        plt.ylabel('Mực nước')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"new_data_prediction_sample_{i}.png")
        print(f"Biểu đồ cho mẫu {i} đã được lưu: new_data_prediction_sample_{i}.png")
        plt.show() 
else: # Nếu không có y_true, chỉ vẽ y_pred
    for i in range(num_samples_to_plot):
        plt.figure(figsize=(12, 6))
        plt.plot(range(P), y_pred_new_np[i, :, 0], label=f'Dự đoán (Mẫu {i})', linestyle='--', marker='x')
        plt.title(f'Dự báo mực nước cho {P} bước tới trên dữ liệu mới (Mẫu {i})')
        plt.xlabel(f'Bước thời gian dự báo (t+1 đến t+{P})')
        plt.ylabel('Mực nước')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"new_data_prediction_only_sample_{i}.png")
        print(f"Biểu đồ dự đoán cho mẫu {i} đã được lưu: new_data_prediction_only_sample_{i}.png")
        plt.show()

print("\nHoàn tất dự đoán trên dữ liệu mới.")
