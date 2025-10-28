import os
import pandas as pd
from datetime import datetime
from ai_history_prediction_strategy.ai_price_predictor import predict_next_price
from data_fetcher import load_close_series


def analyze_growth(tickers, window=31, log_path="ai_predictions.csv"):
    results = []

    for t in tickers:
        s = load_close_series(t)
        if len(s) < window + 2:
            continue

        last = float(s.iloc[-1])
        pred = predict_next_price(s, window)
        if not pred:
            continue

        growth = (pred / last) - 1.0
        results.append({
            "Ticker": t,
            "LastClose": last,
            "PredictedNextClose": pred,
            "PredictedGrowth": growth
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("PredictedGrowth", ascending=False)

    df.insert(0, "Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists(log_path):
        old = pd.read_csv(log_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(log_path, index=False)

    return df


def select_top_n(df, n=10):
    if df.empty:
        return []
    return df.head(n)["Ticker"].tolist()
