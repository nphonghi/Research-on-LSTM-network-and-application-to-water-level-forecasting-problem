# Research on LSTM network and application to water level forecasting problem
The project focuses on researching the LSTM Encoder-Decoder deep learning model, a powerful architecture for time series processing, to address the problem of water level forecasting at the Le Thuy hydrological station in Quang Binh, Vietnam. The water levels are predicted based on historical input data such as water level, rainfall, and tidal information from multiple hydrological stations. The aim is to support effective flood warning and prevention in the context of climate change and the increasing frequency of extreme weather events.
Climate change has significantly increased the frequency and intensity of extreme weather events such as heavy rainfall and storms, posing serious flood risks for Vietnam. Accurate water level forecasting is therefore vital for disaster preparedness and risk management. Traditional time series forecasting models like ARIMA often assume linearity, which limits their effectiveness in capturing complex hydrological dynamics. In contrast, deep learning models, particularly LSTM networks, are capable of learning non-linear relationships and temporal dependencies from data. This project investigates the use of an LSTM Encoder-Decoder architecture, inspired by sequence-to-sequence models in natural language processing, to forecast multi-step water levels at Lệ Thủy station using historical data on water levels, rainfall, and tides.
### Model Comparison: LSTM Encoder-Decoder vs Traditional LSTM

| Metric | Traditional LSTM | LSTM Encoder-Decoder |
|--------|------------------|----------------------|
| MAE    | 0.2985           | **0.1048**           |
| MSE    | 0.3034           | **0.0568**           |
| RMSE   | 0.5508           | **0.2383**           |
| R²     | 0.7072           | **0.9447**           |

As shown, the LSTM Encoder-Decoder model outperformed the traditional LSTM in all key metrics. Notably, the Encoder-Decoder approach demonstrated better generalization for multi-step forecasting, making it more suitable for predicting complex water level patterns under extreme weather conditions.
