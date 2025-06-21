import numpy as np
import pandas as pd

def format_data(filename, attributes, lead_time, predict_time):
    df = pd.read_csv(filename, index_col = False, usecols = attributes, encoding = 'utf-8')[attributes]
    data = np.array(df.values)

    X, y = list(), list()
    for i in range(len(data) - (lead_time + predict_time) + 1):
        x = data[i: i + lead_time, :]
        seq_x = x.reshape(-1)
        X.append(seq_x)

        y_label = data[i + lead_time + predict_time - 1, -1]
        y.append(y_label)
    return np.array(X), np.array(y)