import io
import sys
from src.Traditional.PreprocessData.read_data import format_data
from src.Traditional.Model.model_v1 import Model

# ==================================
# Cấu hình
# ==================================
train_file = 'data/DataPre/Train_WL_2020_2022.csv'
test_file = 'data/DataPre/Test_WL_2023.csv'
attributes = ['WL_KienGiang', 'RF_KienGiang', 'RF_LeThuy', 'WL_DongHoi', 'RF_DongHoi', 'Tide_DongHoi','WL_LeThuy']
lead_time = 24
predict_time = 12

# ==================================
# Tạo tập dữ liệu
# ==================================
X_train, y_train = format_data(train_file, attributes, lead_time, predict_time)
X_test, y_test = format_data(test_file, attributes, lead_time, predict_time)

# ==================================
# Định hình lại dữ liệu 3D cho LSTM
# ==================================
X_train = X_train.reshape((X_train.shape[0], lead_time, len(attributes)))
X_test = X_test.reshape((X_test.shape[0], lead_time, len(attributes)))

# ==================================
# Khởi tạo và huấn luyện mô hình
# ==================================
model = Model(input_shape=(lead_time, len(attributes)), lstm_layers=[10], dense_layers=[50], learning_rate=0.0001)
model.summary()
model.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=20, batch_size=32, verbose=1)

# ==================================
# Dự đoán và đánh giá
# ==================================
y_pred = model.predict(X_test)
metrics = model.evaluate(y_test, y_pred)
print(metrics)

# ==================================
# Lưu mô hình
# ==================================
model.save('models/lstm_model.keras')

# ==================================
# Ghi lại mô hình để lấy dữ liệu so sánh
# ==================================
log_file = "src/Traditional/model_eval_log.txt"

# Tạo buffer để lưu output từ model.summary()
stream = io.StringIO()
sys.stdout = stream
model.summary()
sys.stdout = sys.__stdout__  

summary_str = stream.getvalue()

with open(log_file, 'a') as f:
    f.write(f"\n===== Model Architecture =====\n")
    f.write(summary_str + '\n')

    f.write(f"\nLead time: {lead_time}, Predict time: {predict_time}\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("="*100 + "\n")
