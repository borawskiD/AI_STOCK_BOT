import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime

PORTFOLIO_DIR = "results/portfolios"
RESULTS_DIR = "results"
HISTORY_FILE = os.path.join(RESULTS_DIR, "portfolio_history.csv")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def get_portfolio_value(path):
    if not os.path.exists(path):
        log(f"⚠️ Brak pliku: {path}")
        return None, None, None

    df = pd.read_csv(path)
    if df.empty or "Ticker" not in df.columns or "Shares" not in df.columns:
        log(f"Plik {path} nie zawiera poprawnych danych.")
        return None, None, None

    tickers = df["Ticker"].tolist()
    shares = df["Shares"].tolist()
    prices_now = None

    try:
        data = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            prices_now = data["Close"].iloc[-1]
        else:
            prices_now = data.iloc[-1]
    except Exception as e:
        log(f"Błąd pobierania cen: {e}")
        return None, None, None

    if prices_now is None or len(prices_now) == 0:
        log(f"Brak danych cenowych dla {path}")
        return None, None, None

    current_value = sum(prices_now.get(t, 0) * s for t, s in zip(tickers, shares))
    starting_value = df["CurrentValue($)"].sum() if "CurrentValue($)" in df.columns else None

    return current_value, starting_value, prices_now.to_dict()


def update_history():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(PORTFOLIO_DIR):
        log(f"Brak katalogu {PORTFOLIO_DIR}")
        return

    files = [f for f in os.listdir(PORTFOLIO_DIR) if f.endswith(".csv")]
    if not files:
        log("Brak portfeli do analizy.")
        return

    rows = []
    for fname in files:
        path = os.path.join(PORTFOLIO_DIR, fname)
        current_value, starting_value, prices_dict = get_portfolio_value(path)

        if current_value is None:
            continue

        growth_abs = (current_value - starting_value) if starting_value else 0
        growth_pct = (growth_abs / starting_value * 100) if starting_value else 0

        parts = fname.replace(".csv", "").split("_")
        strategy = parts[0] if len(parts) > 0 else "Unknown"
        tactic = parts[1] if len(parts) > 1 else "None"

        log(f"📊 {fname}: ${current_value:,.2f} ({growth_pct:+.2f}%)")

        rows.append({
            "Timestamp": timestamp,
            "Portfolio": fname,
            "Strategy": strategy,
            "Tactic": tactic,
            "Value($)": current_value,
            "StartValue($)": starting_value,
            "Change($)": growth_abs,
            "Change(%)": growth_pct,
        })

        details_path = os.path.join(RESULTS_DIR, f"history_{strategy}_{tactic}.csv")
        detail_row = {
            "Timestamp": timestamp,
            "Value($)": current_value,
            "Change(%)": growth_pct
        }
        detail_df = pd.DataFrame([detail_row])
        if os.path.exists(details_path):
            old = pd.read_csv(details_path)
            df_all = pd.concat([old, detail_df], ignore_index=True)
        else:
            df_all = detail_df
        df_all.to_csv(details_path, index=False)

        time.sleep(0.3)

    if rows:
        new_df = pd.DataFrame(rows)
        if os.path.exists(HISTORY_FILE):
            old_df = pd.read_csv(HISTORY_FILE)
            df_all = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df_all = new_df
        df_all.to_csv(HISTORY_FILE, index=False)
        log(f"Zaktualizowano historię zbiorczą: {HISTORY_FILE}")

    log("Zakończono aktualizację historii portfeli.")


if __name__ == "__main__":
    log("Start aktualizacji wyników portfeli")
    update_history()
