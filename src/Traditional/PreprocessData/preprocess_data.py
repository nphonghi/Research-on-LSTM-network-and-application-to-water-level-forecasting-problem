import pandas as pd

columns_to_keep = ['Time', 'WL_LongDai', 'WL_HamNinh', 'WL_AnLac',
                   'WL_KienGiang', 'RF_KienGiang', 'WL_LeThuy',
                   'RF_LeThuy', 'WL_DongHoi', 'RF_DongHoi', 'Tide_DongHoi']

def process_and_save_csv(input_path, output_path, time_col='Time'):
    # Đọc dữ liệu
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

    # Chuẩn hóa cột thời gian
    df[datetime_col] = df[datetime_col].astype(str).apply(parse_and_format)

    # Chuẩn hóa các cột còn lại về dạng float với 2 chữ số sau dấu phẩy
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Chuyển thành số
        df[col] = df[col].round(2)  # Làm tròn đến 2 chữ số thập phân

    # Giữ lại các cột cần thiết
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Lưu lại file CSV mới
    df.to_csv(output_path, index=False)

process_and_save_csv('data/DataRaw/Train_WL_2020_2022.csv', 'data/DataPre/Train_WL_2020_2022.csv')
process_and_save_csv('data/DataRaw/Test_WL_2023.csv', 'data/DataPre/Test_WL_2023.csv')
process_and_save_csv('data/DataRaw/Test_WL_2024.csv', 'data/DataPre/Test_WL_2024.csv')