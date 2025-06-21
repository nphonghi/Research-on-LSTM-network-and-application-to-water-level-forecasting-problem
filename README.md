# Research on LSTM network and application to water level forecasting problem.

The project focuses on researching the LSTM Encoder-Decoder deep learning model, a powerful architecture for time series processing, to address the problem of water level forecasting at the Le Thuy hydrological station in Quang Binh, Vietnam. The water levels are predicted based on historical input data such as water level, rainfall, and tidal information from multiple hydrological stations. The aim is to support effective flood warning and prevention in the context of climate change and the increasing frequency of extreme weather events.

Climate change has significantly increased the frequency and intensity of extreme weather events such as heavy rainfall and storms, posing serious flood risks for Vietnam. Accurate water level forecasting is therefore vital for disaster preparedness and risk management. Traditional time series forecasting models like ARIMA often assume linearity, which limits their effectiveness in capturing complex hydrological dynamics. In contrast, deep learning models, particularly LSTM networks, are capable of learning non-linear relationships and temporal dependencies from data. This project investigates the use of an LSTM Encoder-Decoder architecture, inspired by sequence-to-sequence models in natural language processing, to forecast multi-step water levels at Le Thuy station using historical data on water levels, rainfall, and tides.

### 🔍 Model Comparison: LSTM Encoder-Decoder vs Traditional LSTM

| Metric | Traditional LSTM | LSTM Encoder-Decoder |
|--------|------------------|----------------------|
| MAE    | 0.2985           | **0.1048**           |
| MSE    | 0.3034           | **0.0568**           |
| RMSE   | 0.5508           | **0.2383**           |
| R²     | 0.7072           | **0.9447**           |

## 📊 Result Analysis: LSTM Encoder-Decoder vs Traditional LSTM

The results clearly demonstrate the superior performance of the **LSTM Encoder-Decoder** model over the traditional **LSTM** model across all evaluation metrics:

### 🔺 Forecast Errors

- **Mean Squared Error (MSE)**: Decreased significantly from `0.3034` to `0.0568` (**-81.28%**).
- **Mean Absolute Error (MAE)**: Reduced from `0.2985` to `0.1048` (**-64.89%**).
- **Root Mean Squared Error (RMSE)**: Dropped from `0.5508` to `0.2383` (**-56.73%**).

➡️ These improvements indicate that the average prediction error of the proposed Encoder-Decoder model is **approximately half** that of the traditional LSTM.

### 📈 Model Fit (R² Score)

- **Traditional LSTM**: R² = `0.7072` → explains 70.72% of the variance in the actual data.
- **LSTM Encoder-Decoder**: R² = `0.9447` → explains 94.47% of the variance (**+33.58%** improvement).

➡️ This substantial increase in R² shows that the Encoder-Decoder model not only fits the data better but also excels in capturing future trends.

> ✅ **Conclusion**: The LSTM Encoder-Decoder architecture is significantly more accurate and reliable for multi-step water level forecasting, making it a powerful alternative to traditional LSTM-based models.

