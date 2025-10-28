import os
import time
import re
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

PORTFOLIO_DIR = "results/portfolios"
LOG_DIR = "results/update_logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "scheduler_portfolio_log.csv")

REBALANCE_INTERVAL_DAYS = 2
TRIGGER_DROP_THRESHOLD = 0.05  # 5%
SLEEP_BETWEEN = 3


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{msg}\n")


def get_latest_prices(tickers):
    data = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].iloc[-1]
    else:
        prices = data.iloc[-1]
    return prices


def load_portfolio(path):
    try:
        df = pd.read_csv(path)
        if "Ticker" not in df.columns:
            log(f"Brak kolumny 'Ticker' w {path}. Pomijam.")
            return None
        return df
    except Exception as e:
        log(f"Błąd wczytywania {path}: {e}")
        return None


def save_portfolio(df, path):
    df.to_csv(path, index=False)


def update_portfolio(df, prices):
    df["NewPrice"] = df["Ticker"].map(prices)
    df["Change(%)"] = (df["NewPrice"] / df["Price"] - 1) * 100
    df["NewValue($)"] = df["Shares"] * df["NewPrice"]
    total_value = df["NewValue($)"].sum()
    return df, total_value



def trigger_rebalance(df):
    drops = df["Change(%)"] < -TRIGGER_DROP_THRESHOLD * 100
    if drops.any():
        log(f"Spadki > {TRIGGER_DROP_THRESHOLD*100:.1f}% — rebalans wykonany.")
        df["Weight"] = 1 / len(df)
        total_val = df["NewValue($)"].sum()
        df["Investment($)"] = df["Weight"] * total_val
        df["Shares"] = df["Investment($)"] / df["NewPrice"]
    return df


def regular_rebalance(df, path):
    """Rebalans co X dni."""
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime >= timedelta(days=REBALANCE_INTERVAL_DAYS):
        log(f"🔄 Minęło {REBALANCE_INTERVAL_DAYS} dni — rebalans wykonany.")
        total_val = df["NewValue($)"].sum()
        df["Weight"] = 1 / len(df)
        df["Investment($)"] = df["Weight"] * total_val
        df["Shares"] = df["Investment($)"] / df["NewPrice"]
    return df


def static_rebalance(df):
    log("Tryb statyczny — brak rebalansu.")
    return df


# === GŁÓWNY SCHEDULER ===

def run_scheduler():
    log("Start aktualizacji wszystkich portfeli\n")

    files = [f for f in os.listdir(PORTFOLIO_DIR) if f.endswith(".csv")]
    if not files:
        log("Brak portfeli w katalogu.")
        return

    for fname in files:
        path = os.path.join(PORTFOLIO_DIR, fname)
        match = re.match(r"([A-Z]+)_([A-Z]+)\.csv", fname)
        if not match:
            log(f"Nie rozpoznano wzorca nazwy pliku: {fname}")
            continue

        strategy, tactic = match.groups()
        log(f"\n Aktualizacja portfela: {fname} | Strategia={strategy}, Taktyka={tactic}")

        df = load_portfolio(path)
        if df is None or df.empty:
            continue

        tickers = df["Ticker"].tolist()
        prices = get_latest_prices(tickers)
        df, total_value = update_portfolio(df, prices)

        if "TRIGGER" in tactic:
            df = trigger_rebalance(df)
        elif "REGULAR" in tactic:
            df = regular_rebalance(df, path)
        elif "STATIC" in tactic:
            df = static_rebalance(df)
        else:
            log(f"Nieznana taktyka: {tactic}")
            continue

        # Zapis
        save_portfolio(df, path)
        total_change = (df["NewValue($)"].sum() / df["CurrentValue($)"].sum() - 1) * 100
        log(f"Zaktualizowano {fname} | Δ {total_change:+.2f}% | Wartość=${total_value:,.2f}\n")

        time.sleep(SLEEP_BETWEEN)

    log("Zakończono harmonogram aktualizacji wszystkich portfeli.\n")


if __name__ == "__main__":
    run_scheduler()
