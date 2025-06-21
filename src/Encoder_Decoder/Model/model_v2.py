import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==================================
# Lớp Encoder
# ==================================
class Encoder(layers.Layer):
    """
    Lớp Encoder LSTM.

    Nhận chuỗi đầu vào là dữ liệu lịch sử và tạo ra các trạng thái ẩn(h)/ô cuối cùng(c) từ tất cả các lớp LSTM để truyền cho Decoder.
    """
    def __init__(self, features, lstm_units, num_layers, name='encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.features = features
        self.lstm_units = lstm_units
        self.num_layers = num_layers

        self.lstm_layers = []
        for i in range(num_layers):
            # TẤT CẢ các lớp cần return_state=True để lấy trạng thái cuối.
            # TẤT CẢ các lớp trừ lớp cuối cùng cần return_sequences=True để truyền chuỗi đầy đủ cho lớp tiếp theo.
            is_last_layer = (i == num_layers - 1)
            self.lstm_layers.append(layers.LSTM(
                lstm_units,
                return_sequences=not is_last_layer,
                return_state=True,
                name=f'encoder_lstm_{i}'
            ))

    def call(self, inputs):
        """
        Xử lý chuỗi đầu vào và trả về trạng thái từ tất cả các lớp.

        Args:
            inputs: Tensor đầu vào lịch sử, shape (batch_size, T, features)

        Returns:
            encoder_output: Đầu ra của lớp LSTM cuối cùng.
                            Shape (batch_size, lstm_units).
            encoder_states: List các cặp trạng thái [hidden_state, cell_state] từ TẤT CẢ các lớp LSTM tại bước thời gian cuối cùng.
                            Shape: [[h0, c0], [h1, c1], ..., [hN-1, cN-1]].
        """

        encoder_hiddens = []

        # INPUT [batch_size, T, features]
        x = inputs
        for layer in self.lstm_layers:
            """
            Mỗi lớp LSTM trả về:
                                output: OUTPUT [batch_size, T, lstm_units],
                                encoder_hidden
            """
            x, state_h, state_c = layer(x)
            encoder_hiddens.append([state_h, state_c]) # Lưu trạng thái [h, c] của lớp này

        # x là đầu ra của lớp cuối cùng
        # Ở đây, x sẽ là (batch_size, lstm_units) vì lớp cuối có return_sequences=False
        # ENCODER_OUTPUT: [batch_size, 1, lstm_units]
        encoder_output = x
        return encoder_output, encoder_hiddens

    def get_config(self):
        config = super().get_config()
        config.update({
            'features': self.features,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
        })
        return config

# ==================================
# Lớp Decoder
# ==================================
class Decoder(layers.Layer):
    """
    Lớp Decoder LSTM.

    Nhận trạng thái từ Encoder làm trạng thái khởi tạo.
    """
    def __init__(self, output_dim, lstm_units, num_layers, name='decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers

        self.lstm_layers = []
        for i in range(num_layers):
            # TẤT CẢ các lớp LSTM trong Decoder cần return_sequences=True để có thể áp dụng lớp Dense lên đầu ra tại mỗi bước (ở đây chỉ xử lý 1 bước thời gian).
            # TẤT CẢ các lớp cần return_state=True để lấy trạng thái mới và truyền cho bước thời gian tiếp theo của Decoder.
            self.lstm_layers.append(layers.LSTM(
                lstm_units,
                return_sequences=True,
                return_state=True,
                name=f'decoder_lstm_{i}'
            ))

        # Lớp Dense để chuyển đổi đầu ra LSTM thành dự đoán cuối cùng
        self.dense_output = layers.Dense(output_dim, name='decoder_output')

    def call(self, inputs, initial_states):
        """
        Xử lý một bước giải mã.

        Args:
            inputs: Đầu vào cho bước thời gian hiện tại của decoder.
                    Shape (batch_size, 1, feature), feature là output_dim.
            initial_states: List các trạng thái [hidden_state, cell_state] từ Encoder hoặc từ bước thời gian trước của Decoder.
                            [[h0, c0], [h1, c1], ..., [hN-1, cN-1]]

        Returns:
            prediction: Dự đoán cho bước thời gian hiện tại. Shape (batch_size, 1, output_dim).
            decoder_states: List các trạng thái [hidden_state, cell_state] mới từ các lớp LSTM
                            của decoder sau bước này. Dùng làm initial_states cho bước tiếp theo.
        """
        if not isinstance(initial_states, list) or len(initial_states) != self.num_layers:
             raise ValueError(f"initial_states phải là list gồm {self.num_layers} cặp [h, c]. "
                              f"Nhận được: {type(initial_states)} với độ dài {len(initial_states) if isinstance(initial_states, list) else 'N/A'}")

        x = inputs
        decoder_states_new = [] # Lưu trữ trạng thái mới từ mỗi lớp

        for i, layer in enumerate(self.lstm_layers):
            # layer trả về output_sequence, state_h, state_c
            x, state_h, state_c = layer(x, initial_state=initial_states[i])
            decoder_states_new.append([state_h, state_c]) # Lưu trạng thái mới

        # x bây giờ là đầu ra của lớp LSTM cuối cùng, shape (batch_size, 1, lstm_units)
        # Áp dụng lớp Dense để tạo dự đoán
        prediction = self.dense_output(x) # Shape (batch_size, 1, output_dim)

        return prediction, decoder_states_new

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
        })
        return config

# ==================================
# Lớp Mô hình Encoder-Decoder
# ==================================
class EncoderDecoderModel(keras.Model):
    """
    Mô hình LSTM Encoder-Decoder cho dự báo chuỗi thời gian.
    """
    def __init__(self, input_dim, output_dim, lstm_units, num_layers, prediction_length,
                 name='encoder_decoder_model', **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.prediction_length = prediction_length # Số bước dự đoán P

        self.encoder = Encoder(input_dim, lstm_units, num_layers)
        self.decoder = Decoder(output_dim, lstm_units, num_layers)

        self.encoder_transformer = layers.Dense(output_dim, name='encoder_transformer')

    def call(self, inputs, training=None):
        """
        Thực hiện quá trình dự báo.

        Args:
            inputs (dict): Một dictionary chứa:
                'encoder_inputs': Dữ liệu lịch sử cho encoder.
                                  Shape (batch_size, T, input_dim).
                'initial_decoder_input': Đầu vào cho decoder.
                                         Shape (batch_size, 1, output_dim).
            training (bool): Cờ cho biết đang ở chế độ huấn luyện hay không (từ Keras).
            use_teacher_forcing (bool): Nếu True và teacher_forcing_targets được cung cấp, sử dụng giá trị thực làm đầu vào cho bước tiếp theo của decoder.
                                        Ngược lại, sử dụng dự đoán của mô hình (autoregressive).
            teacher_forcing_targets (Tensor, optional): Chuỗi mục tiêu thực tế để sử dụng cho teacher forcing.
                                                        Shape (batch_size, P, output_dim).

        Returns:
            all_predictions: Tensor chứa các dự đoán cho P bước thời gian trong tương lai.
                             Shape (batch_size, P, output_dim).
        """
        encoder_inputs = inputs['encoder_inputs']
        teacher_forcing_targets = inputs.get('teacher_forcing_targets', None)

        # 1. Chạy Encoder để lấy trạng thái ngữ cảnh
        encoder_output, encoder_states = self.encoder(encoder_inputs)
        encoder_output = layers.Reshape((1, self.lstm_units))(encoder_output)
        initial_decoder_input = self.encoder_transformer(encoder_output)

        # 2. Khởi tạo vòng lặp Decoder
        all_predictions = []
        decoder_input_current = initial_decoder_input # Đầu vào đầu tiên là đầu ra của encoder
        decoder_states_current = encoder_states # Trạng thái ban đầu từ encoder

        for i in range(self.prediction_length):
            # Chạy một bước giải mã
            prediction, decoder_states_new = self.decoder(
                decoder_input_current, initial_states=decoder_states_current
            )
            # prediction shape: (batch_size, 1, output_dim)
            # decoder_states_new shape: [[h0, c0], [h1, c1], ...]

            # Lưu trữ dự đoán
            all_predictions.append(prediction)

            # Cập nhật trạng thái cho bước tiếp theo
            decoder_states_current = decoder_states_new

            # Chuẩn bị đầu vào cho bước tiếp theo
            if training and teacher_forcing_targets is not None:
                decoder_input_current = teacher_forcing_targets[:, i:i+1, :]
            else:
                # Sử dụng dự đoán của mô hình làm đầu vào tiếp theo (autoregressive)
                decoder_input_current = prediction

        # Kết hợp tất cả các dự đoán thành một tensor duy nhất
        # all_predictions là list các tensor shape (batch, 1, output_dim)
        final_predictions = tf.concat(all_predictions, axis=1) # Shape (batch, P, output_dim)

        return final_predictions

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'prediction_length': self.prediction_length,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
