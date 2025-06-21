import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

class Model:
    def __init__(self, input_shape, lstm_layers=[100], dense_layers=[50, 30, 20], learning_rate=0.001):
        self.input_shape = input_shape
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        for i, units in enumerate(self.lstm_layers):
            return_seq = i < len(self.lstm_layers) - 1  # Chỉ lớp LSTM cuối cùng không trả về chuỗi
            model.add(LSTM(units, activation='tanh', 
                           return_sequences=return_seq,
                           kernel_initializer='he_normal', 
                           input_shape=self.input_shape if i == 0 else None))

        for units in self.dense_layers:
            model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))

        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32, verbose=1):
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val) if X_val is not None else None,
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose)

    def predict(self, X):
        return self.model.predict(X).reshape(-1)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    def save(self, filepath):
        self.model.save(filepath)

    def summary(self):
        self.model.summary()