import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def _prepare_features(series, window=30, horizon=1):
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= window + horizon:
        raise ValueError(
            f"Za mało danych ({len(arr)} punktów) do utworzenia okien "
            f"o długości {window} i horyzoncie {horizon}."
        )

    X, y = [], []

    for i in range(window, len(arr) - horizon + 1):
        window_slice = arr[i - window:i]
        target_value = arr[i + horizon - 1]
        X.append(window_slice)
        y.append(target_value)

    X, y = np.array(X), np.array(y)

    if np.isnan(X).any() or np.isnan(y).any():
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

    return X, y


def predict_next_price(close_series, window=5):

    if len(close_series) < window + 2:
        return None

    X, y = _prepare_features(close_series, window, )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xs, y)

    last_window = np.asarray(close_series[-window:], dtype=float).reshape(1, -1)
    predicted = model.predict(scaler.transform(last_window))[0]
    return float(predicted)
