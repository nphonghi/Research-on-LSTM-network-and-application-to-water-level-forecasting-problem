from src.Traditional.PreprocessData.read_data import format_data
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Đọc dữ liệu từ file và chuyển thành tập các chuỗi
new_file = 'data/DataPre/Test_WL_2024.csv'
attributes = ['WL_KienGiang', 'RF_KienGiang', 'RF_LeThuy', 'WL_DongHoi', 'RF_DongHoi', 'Tide_DongHoi','WL_LeThuy']
lead_time = 24
predict_time = 12

# Tạo tập dữ liệu
X, y_real = format_data(new_file, attributes, lead_time, predict_time)

# Chuyển dữ liệu thành ma trận 3 chiều [samples, leadtime, attributes]
X = X.reshape((X.shape[0], lead_time, len(attributes)))

# Load model để dự báo
model = load_model('models/lstm_model.keras')

# Dự báo nhãn của X
y_pred = model.predict(X)
y_pred = y_pred.reshape(y_pred.shape[0])

# Ma trận đánh giá
def evaluate(y_real, y_pred):
    mse = mean_squared_error(y_real, y_pred)
    rmse = root_mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

metrics = evaluate(y_real, y_pred)
print(metrics)

# Lưu kết quả dự báo của mô hình
log_file = "src/Traditional/model_test_log.txt"

with open(log_file, 'a') as f:
    f.write(f"\nLead time: {lead_time}, Predict time: {predict_time}\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("="*100 + "\n")

# Vẽ biểu đồ so sánh y_test và y_pred
def plot_prediction(y_real, y_pred, timestamps=None, station_name="Le Thuy", prediction_time=3):
    plt.figure(figsize=(14, 6))
    
    # Vẽ đường dự báo và thực tế
    plt.plot(y_real, label='Observed (y_real)', color='black', linewidth=2)
    plt.plot(y_pred, label='Predicted (y_pred)', color='red', linestyle='--', linewidth=2)

    plt.title(f'{station_name} Station\nPrediction Horizon = {prediction_time} hours', fontsize=16, fontweight='bold')
    plt.xlabel('Time' if timestamps is None else 'Timestamp', fontsize=12)
    plt.ylabel('Water Level (m)', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    
    if timestamps is not None:
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig("src/Traditional/Images/my_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

plot_prediction(y_real, y_pred)
