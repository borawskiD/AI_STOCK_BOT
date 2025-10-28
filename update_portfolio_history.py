import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime

RESULTS_DIR = "results"
PORTFOLIOS = {
    "AI": "portfolio_ai.csv",
    "News": "portfolio_news.csv",
    "Random": "portfolio_random.csv"
}
HISTORY_FILE = os.path.join(RESULTS_DIR, "portfolio_history.csv")


def get_portfolio_value(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    tickers = df["Ticker"].tolist()
    shares = df["Shares"].tolist()
    data = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].iloc[-1]
    else:
        prices = data["Close"].iloc[-1]
    current_value = sum(prices[t] * s for t, s in zip(tickers, shares) if t in prices)
    return current_value


def update_history():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for name, file in PORTFOLIOS.items():
        path = os.path.join(RESULTS_DIR, file)
        value = get_portfolio_value(path)
        if value is not None:
            print(f"{name}: {value:,.2f}")
            rows.append({"Timestamp": timestamp, "Strategy": name, "Value": value})
        else:
            print(f"Brak danych dla {name}")

    if rows:
        new_df = pd.DataFrame(rows)
        if os.path.exists(HISTORY_FILE):
            old_df = pd.read_csv(HISTORY_FILE)
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_csv(HISTORY_FILE, index=False)
        print(f"Zaktualizowano historię: {HISTORY_FILE}")


if __name__ == "__main__":
    print("Aktualizacja historii portfeli...")
    update_history()
