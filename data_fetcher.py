# data_fetcher.py
import time
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/prices")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _fresh_enough(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    age_sec = time.time() - path.stat().st_mtime
    return age_sec <= max_age_hours * 3600

def fetch_price_history(tickers, period="6mo", overwrite=False, max_age_hours=24):
    to_fetch = []
    for t in tickers:
        path = DATA_DIR / f"{t}.csv"
        if not overwrite and _fresh_enough(path, max_age_hours):
            continue
        to_fetch.append(t)

    if not to_fetch:
        print("Dane historyczne są aktualne — pomijam pobieranie.")
        return

    print(f"Pobieram dane dla {len(to_fetch)} spółek (period={period})...")
    for i, t in enumerate(to_fetch, start=1):
        try:
            df = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                df.to_csv(DATA_DIR / f"{t}.csv")
                print(f"  [{i}/{len(to_fetch)}] ✅ {t}")
        except Exception as e:
            print(f"⚠️ Błąd pobierania {t}: {e}")
        time.sleep(0.2)
    print("Pobieranie danych zakończone.")


DATA_DIR = Path("data/prices")

def load_close_series(ticker):
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        print(f"Brak danych dla {ticker}")
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(path)
        if "Close" in df.columns:
            s = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(s) > 0:
                return s
    except Exception:
        pass

    try:
        df = pd.read_csv(path, skiprows=3)
        if "Close" in df.columns:
            df = df.rename(columns={"Price": "Date"})
            s = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(s) > 0:
                return s
    except Exception as e:
        print(f"Nie udało się odczytać {ticker}: {e}")

    print(f"Nie udało się znaleźć danych Close dla {ticker}")
    return pd.Series(dtype=float)