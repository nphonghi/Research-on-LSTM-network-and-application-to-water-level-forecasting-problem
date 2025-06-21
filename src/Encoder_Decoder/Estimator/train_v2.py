import io
import sys
import pandas as pd
import numpy as np
import joblib
from src.Encoder_Decoder.PreprocessData.prepare_data import prepare_data 
from src.Encoder_Decoder.Model.model_v2 import EncoderDecoderModel 
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

train_file = 'data/DataScaled/Train_WL_2020_2022.csv'
test_file = 'data/DataScaled/Test_WL_2023.csv'
attributes = ['WL_KienGiang', 'WL_DongHoi','WL_LeThuy', 'RF_DongHoi', 'Tide_DongHoi', 'RF_KienGiang', 'RF_LeThuy']
target_attribute = 'WL_LeThuy' 

T = 24  # lead_time: số bước quá khứ
P = 12  # predict_time: số bước dự báo
n_input_features = len(attributes) 
n_output_features = 1 # Chỉ dự báo mực nước
lstm_hidden_units = 64
n_lstm_layers = 3

USE_TEACHER_FORCING_DURING_TRAINING = False

# --- Tạo tập dữ liệu ---
encoder_inputs_train, decoder_targets_train = prepare_data(
    train_file, attributes, T, P, target_attribute
)
encoder_inputs_test, decoder_targets_test = prepare_data(
    test_file, attributes, T, P, target_attribute
)

# Dữ liệu mục tiêu huấn luyện (y_true)
y_train_true = tf.convert_to_tensor(decoder_targets_train, dtype=tf.float32)
y_test_true = tf.convert_to_tensor(decoder_targets_test, dtype=tf.float32)

# --- Chuẩn bị dữ liệu đầu vào cho phương thức `call` của mô hình ---
# Dữ liệu huấn luyện
train_input_dict = {
    'encoder_inputs': tf.convert_to_tensor(encoder_inputs_train, dtype=tf.float32)
}

if USE_TEACHER_FORCING_DURING_TRAINING:
    train_input_dict['teacher_forcing_targets'] = y_train_true
    print("Teacher forcing sẽ được sử dụng trong quá trình huấn luyện.")
else:
    print("Teacher forcing sẽ KHÔNG được sử dụng trong quá trình huấn luyện.")

# Dữ liệu kiểm thử
test_input_dict = {
    'encoder_inputs': tf.convert_to_tensor(encoder_inputs_test, dtype=tf.float32)
}

# --- Khởi tạo mô hình Encoder-Decoder ---
enc_dec_model = EncoderDecoderModel(
    input_dim=n_input_features,
    output_dim=n_output_features,
    lstm_units=lstm_hidden_units,
    num_layers=n_lstm_layers,
    prediction_length=P
)

# Compile mô hình
enc_dec_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='mse', # Mean Squared Error
    metrics=['mae'] # Mean Absolute Error
) 

# Summary
_ = enc_dec_model(train_input_dict)
enc_dec_model.summary()

print("Bắt đầu huấn luyện...")
history = enc_dec_model.fit(
    train_input_dict,
    y_train_true,         
    epochs=20,
    batch_size=32,
    validation_data=(test_input_dict, y_test_true), 
    verbose=1
)

# Dự đoán 
print("Bắt đầu dự đoán trên tập test...")

y_pred_sequences_tf = enc_dec_model.predict(test_input_dict) # Đây là tensor của TensorFlow
y_pred_sequences_np = y_pred_sequences_tf

# Đánh giá 
print("\nĐánh giá mô hình trên tập Test:")

# Chuyển y_test_true sang NumPy array
y_test_true_np = y_test_true.numpy()

# Reverse kết quả
scaler = joblib.load('data/DataScaled/minmax_scaler2.pkl')

shape = y_pred_sequences_np.shape  
y_pred_2d = y_pred_sequences_np.reshape(-1, shape[-1])  # [batch_size * timesteps, n_features]
y_pred_2d_original = scaler.inverse_transform(y_pred_2d)
y_pred_sequences_np = y_pred_2d_original.reshape(shape)

shape = y_test_true_np.shape  
y_true_2d = y_test_true_np.reshape(-1, shape[-1])  # [batch_size * timesteps, n_features]
y_true_2d_original = scaler.inverse_transform(y_true_2d)
y_test_true_np = y_true_2d_original.reshape(shape)

# Để tính toán các metrics tổng thể, flatten chuỗi dự đoán và chuỗi thực tế
# vì các hàm của sklearn thường mong đợi vector 1D.
y_test_true_flat = y_test_true_np.reshape(-1)
y_pred_sequences_flat = y_pred_sequences_np.reshape(-1)

# Tính toán và in các metrics 
mse_overall = mean_squared_error(y_test_true_flat, y_pred_sequences_flat)
rmse_overall = root_mean_squared_error(y_test_true_flat, y_pred_sequences_flat)
mape_overall = mean_absolute_percentage_error(y_test_true_flat, y_pred_sequences_flat)
mae_overall = mean_absolute_error(y_test_true_flat, y_pred_sequences_flat)
r2_overall = r2_score(y_test_true_flat, y_pred_sequences_flat)

print(f"Overall MSE: {mse_overall:.4f}")
print(f"Overall RMSE: {rmse_overall:.4f}")
print(f"Overall MAE: {mae_overall:.4f}") 
print(f"Overall MAPE: {mape_overall:.4f}") 
print(f"Overall R2 Score: {r2_overall:.4f}")

# Đánh giá tại từng bước dự đoán (0 đến P-1)
print("\nĐánh giá tại từng bước dự báo (horizon):")
for step in range(P):
    y_true_step = y_test_true_np[:, step, 0]
    y_pred_step = y_pred_sequences_np[:, step, 0]
    
    mse_step = mean_squared_error(y_true_step, y_pred_step)
    rmse_step = root_mean_squared_error(y_true_step, y_pred_step)
    mape_step = mean_absolute_percentage_error(y_true_step, y_pred_step)
    mae_step = mean_absolute_error(y_true_step, y_pred_step)
    r2_step = r2_score(y_true_step, y_pred_step)
    
    print(f"  Horizon t+{step+1}: RMSE={rmse_step:.4f}, MAE={mae_step:.4f}, MAPE={mape_step:.4f}, R2={r2_step:.4f}")


# --- Lưu mô hình ---
model_save_path = 'models/encoder_decoder_lstm_model_v2.keras'
enc_dec_model.save(model_save_path)
print(f"Mô hình đã được lưu tại: {model_save_path}")


sample_index = 0 
plt.figure(figsize=(12, 6))
plt.plot(range(P), y_test_true_np[sample_index, :, 0], label='Thực tế (y_true)', marker='o')
plt.plot(range(P), y_pred_sequences_np[sample_index, :, 0], label='Dự đoán (y_pred)', linestyle='--', marker='x')
plt.title(f'Dự báo mực nước cho {P} bước tới (Mẫu {sample_index})')
plt.xlabel(f'Bước thời gian dự báo (t+1 đến t+{P})')
plt.ylabel('Mực nước')
plt.legend()
plt.grid(True)
plt.savefig("src/Encoder_Decoder/Images/encoder_decoder_prediction_sample_v2.png")
plt.show()

# Tính giá trị trung bình theo từng bước dự báo cho toàn bộ mẫu
mean_true = y_test_true_np[:, :, 0].mean(axis=0)
mean_pred = y_pred_sequences_np[:, :, 0].mean(axis=0)

plt.figure(figsize=(12, 6))
plt.plot(range(P), mean_true, label='Giá trị thực trung bình', marker='o')
plt.plot(range(P), mean_pred, label='Giá trị dự đoán trung bình', linestyle='--', marker='x')
plt.title(f'Trung bình giá trị thực và dự đoán trên tất cả {y_test_true_np.shape[0]} mẫu')
plt.xlabel(f'Bước thời gian dự báo (t+1 đến t+{P})')
plt.ylabel('Mực nước')
plt.legend()
plt.grid(True)
plt.savefig("src/Encoder_Decoder/Images/average_all_samples.png")
plt.show()

plt.figure(figsize=(14, 7))
for i in range(y_test_true_np.shape[0]):
    plt.plot(range(P), y_test_true_np[i, :, 0], color='blue', alpha=0.05)  # Thực tế
    plt.plot(range(P), y_pred_sequences_np[i, :, 0], color='red', alpha=0.1)  # Dự đoán

plt.title(f'Giá trị thực và dự đoán cho toàn bộ {y_test_true_np.shape[0]} mẫu')
plt.xlabel(f'Bước thời gian dự báo (t+1 đến t+{P})')
plt.ylabel('Mực nước')
plt.grid(True)
plt.savefig("src/Encoder_Decoder/Images/overlay_all_samples.png")
plt.show()

batch_size = 32
cols = 8
rows = batch_size // cols

plt.figure(figsize=(20, 10))
for i in range(batch_size):
    plt.subplot(rows, cols, i + 1)
    plt.plot(range(P), y_test_true_np[i, :, 0], label='Thực tế', marker='o', linewidth=1)
    plt.plot(range(P), y_pred_sequences_np[i, :, 0], label='Dự đoán', linestyle='--', marker='x', linewidth=1)
    plt.title(f'Mẫu {i}')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.suptitle('Dự báo trên mini-batch 32 mẫu đầu tiên', fontsize=16, y=1.02)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.savefig("src/Encoder_Decoder/Images/batch_32_predictions.png", bbox_inches='tight')
plt.show()

# ĐỊNH NGHĨA METRICS 
metrics_overall = {
    "Overall MSE": mse_overall,
    "Overall RMSE": rmse_overall,
    "Overall MAE": mae_overall,
    "Overall R2 Score": r2_overall
}

metrics_horizon = {}
for step in range(P): 
    y_true_step = y_test_true_np[:, step, 0] 
    y_pred_step = y_pred_sequences_np[:, step, 0]
    
    mse_step = mean_squared_error(y_true_step, y_pred_step)
    rmse_step = root_mean_squared_error(y_true_step, y_pred_step)
    mape_step = mean_absolute_percentage_error(y_true_step, y_pred_step)
    mae_step = mean_absolute_error(y_true_step, y_pred_step)
    r2_step = r2_score(y_true_step, y_pred_step)
    
    metrics_horizon[f"Horizon t+{step+1} MSE"] = mse_step
    metrics_horizon[f"Horizon t+{step+1} RMSE"] = rmse_step
    metrics_horizon[f"Horizon t+{step+1} MAE"] = mae_step
    metrics_horizon[f"Horizon t+{step+1} R2 Score"] = r2_step

all_metrics_to_log = {**metrics_overall, **metrics_horizon}

log_file = "src/Encoder_Decoder/model_eval_log_v2.txt"

# Tạo buffer để lưu output từ model.summary()
stream = io.StringIO()

sys.stdout = stream
enc_dec_model.summary(expand_nested=True, show_trainable=True)
sys.stdout = sys.__stdout__  

summary_str = stream.getvalue()

with open(log_file, 'a') as f:
    f.write(f"\n===== Model Evaluation Log - {pd.Timestamp.now()} =====\n") 
    f.write(f"\n----- Model Architecture -----\n")
    f.write(summary_str + '\n')

    f.write(f"----- Configuration -----\n")
    f.write(f"Lead time (T): {T}\n")
    f.write(f"Predict time (P): {P}\n")
    f.write(f"LSTM Units: {lstm_hidden_units if 'lstm_hidden_units' in locals() else 'N/A'}\n") 
    f.write(f"LSTM Layers: {n_lstm_layers if 'n_lstm_layers' in locals() else 'N/A'}\n") 
    if 'USE_TEACHER_FORCING_DURING_TRAINING' in locals() and 'history' in locals() :
        f.write(f"Teacher Forcing during training: {USE_TEACHER_FORCING_DURING_TRAINING}\n")
        f.write(f"Epochs trained: {len(history.history['loss']) if history else 'N/A'}\n")

    f.write(f"\n----- Overall Performance Metrics -----\n")
    for k, v in metrics_overall.items(): 
        f.write(f"{k}: {v:.4f}\n")
    
    if metrics_horizon: 
        f.write(f"\n----- Performance Metrics by Horizon -----\n")
        for k, v in metrics_horizon.items():
            f.write(f"{k}: {v:.4f}\n")
            
    f.write("="*80 + "\n")

print(f"Log đã được ghi vào file: {log_file}")
