import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

columns_to_keep = ['Time', 'WL_KienGiang', 'WL_LeThuy', 'WL_DongHoi', 'RF_DongHoi', 'Tide_DongHoi', 'RF_KienGiang', 'RF_LeThuy']
value_columns = [col for col in columns_to_keep if col != 'Time']


# ==================================
# Hàm xử lý đầu vào
# ==================================
def process_csv_for_scaling(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    datetime_col = df.columns[0]

    def parse_and_format(val):
        try:
            dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
            if pd.isnull(dt):
                return None
            return dt.strftime('%d/%m/%Y %H:%M')
        except:
            return None

    # Xử lý thời gian
    df[datetime_col] = df[datetime_col].astype(str).apply(parse_and_format)

    # Chuyển các cột số thành float
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

    # Giữ lại các cột cần thiết
    df = df[[col for col in columns_to_keep if col in df.columns]]

    return df

# Xử lý 3 tập
train_df = process_csv_for_scaling('Data/Train_WL_2020_2022.csv')
val_df   = process_csv_for_scaling('Data/Test_WL_2023.csv')
test_df  = process_csv_for_scaling('Data/Test_WL_2024.csv')

# Tiền xử lý dữ liệu
scaler = MinMaxScaler()
train_scaled_values = scaler.fit_transform(train_df[value_columns])
val_scaled_values   = scaler.transform(val_df[value_columns])
test_scaled_values  = scaler.transform(test_df[value_columns])

# Lưu scaler để dùng sau
joblib.dump(scaler, 'DataScaled/minmax_scaler.pkl')

# Scale riêng cho biến mục tiêu để reverse kết quả
scaler2 = MinMaxScaler()
train_scaled_target = scaler2.fit_transform(train_df['WL_LeThuy'].values.reshape(-1, 1))
val_scaled_target   = scaler2.transform(val_df['WL_LeThuy'].values.reshape(-1, 1))
test_scaled_target  = scaler2.transform(test_df['WL_LeThuy'].values.reshape(-1, 1))

joblib.dump(scaler2, 'DataScaled/minmax_scaler2.pkl')

# Gộp lại với cột Time
train_scaled_df = pd.concat([train_df[['Time']], pd.DataFrame(train_scaled_values, columns=value_columns)], axis=1)
val_scaled_df   = pd.concat([val_df[['Time']],   pd.DataFrame(val_scaled_values,   columns=value_columns)], axis=1)
test_scaled_df  = pd.concat([test_df[['Time']],  pd.DataFrame(test_scaled_values,  columns=value_columns)], axis=1)

# Lưu kết quả ra CSV
os.makedirs('DataScaled', exist_ok=True)
train_scaled_df.to_csv('DataScaled/Train_WL_2020_2022.csv', index=False)
val_scaled_df.to_csv('DataScaled/Test_WL_2023.csv', index=False)
test_scaled_df.to_csv('DataScaled/Test_WL_2024.csv', index=False)
