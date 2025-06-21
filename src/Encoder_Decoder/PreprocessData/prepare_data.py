import numpy as np
import pandas as pd

def prepare_data(filename, attributes, T, P, target_attribute):
    """
    Chuẩn bị dữ liệu cho mô hình LSTM: Encoder-Decoder.

    Args:
        filename (str): Đường dẫn đến file CSV.
        attributes (list): Danh sách tên các cột thuộc tính để sử dụng.
        T (int): Số bước thời gian quá khứ (lookback window, lead_time).
        P (int): Số bước thời gian dự báo tương lai (prediction horizon, predict_time).
        target_attribute (str): Tên của cột chứa giá trị mục tiêu cần dự báo.

    Returns:
        tuple: Gồm 3 numpy arrays:
            - encoder_inputs: (num_samples, T, num_input_features)
            - decoder_initial_inputs: (num_samples, 1, 1) - Giá trị mục tiêu tại t
            - decoder_targets: (num_samples, P, target_attribute) - Chuỗi giá trị mục tiêu từ t đến t+P
    """
    df = pd.read_csv(filename, index_col=False, usecols=attributes, encoding='utf-8')[attributes]
    data = np.array(df.values) # Dữ liệu dạng numpy array (num_timesteps, num_attributes)

    # Xác định index của cột mục tiêu
    try:
        target_col_index = attributes.index(target_attribute)
    except ValueError:
        raise ValueError(f"Target attribute '{target_attribute}' not found in attributes list.")

    num_timesteps = data.shape[0]
    num_input_features = data.shape[1]

    encoder_input = list()
    decoder_initial_input = list()
    decoder_target = list()

    # Tạo cửa sổ dữ liệu trượt
    for i in range(num_timesteps - (T + P) + 1):
        # 1. Encoder Inputs: T bước quá khứ, tất cả các thuộc tính
        input = data[i : i + T, :]
        encoder_input.append(input)

        # 2. Decoder Initial Input: Giá trị mục tiêu tại bước cuối cùng của chuỗi encoder (tức t)
        # Lấy giá trị từ cột mục tiêu tại index i + T - 1
        initial_decoder_input_val = data[i + T - 1, target_col_index]
        # Reshape thành (1, 1) để phù hợp với input decoder (batch_size, 1, output_dim=1)
        decoder_initial_input.append(initial_decoder_input_val.reshape(1, 1))

        # 3. Decoder Targets: P bước tương lai, chỉ cột mục tiêu
        # Lấy P giá trị từ cột mục tiêu, bắt đầu từ index i + T
        target_seq = data[i + T : i + T + P, target_col_index]
        # Reshape thành (P, 1) để phù hợp với output decoder (batch_size, P, output_dim=1)
        decoder_target.append(target_seq.reshape(P, 1))

    # Chuyển list các numpy array thành các numpy array lớn
    encoder_inputs = np.array(encoder_input)
    decoder_initial_inputs = np.array(decoder_initial_input)
    decoder_targets = np.array(decoder_target)

    return encoder_inputs, decoder_targets