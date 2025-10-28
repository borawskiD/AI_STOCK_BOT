# ai_trend_strategy_remote.py
import os, time, requests, yfinance as yf, pandas as pd
from pathlib import Path
from datetime import datetime

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
DATA_DIR = Path("../data/prices")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_price_history(tickers, period="3mo", overwrite=False):

    to_fetch = []
    for t in tickers:
        path = DATA_DIR / f"{t}.csv"
        if not overwrite and path.exists():
            continue
        to_fetch.append(t)

    if not to_fetch:
        return

    for i, t in enumerate(to_fetch, start=1):
        try:
            df = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                df.to_csv(DATA_DIR / f"{t}.csv")
                print(f"  [{i}/{len(to_fetch)}] ✅ {t}")
            time.sleep(0.3)
        except Exception as e:
            print(f"Błąd pobierania {t}: {e}")


def summarize_stock_trend(ticker):
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "Close" not in df.columns:
        return None

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    if len(df) < 10:
        return None

    days_30 = min(30, len(df) - 1)
    days_7 = min(7, len(df) - 1)

    change_30 = (df["Close"].iloc[-1] / df["Close"].iloc[-days_30]) - 1
    change_7 = (df["Close"].iloc[-1] / df["Close"].iloc[-days_7]) - 1

    desc = (
        f"The stock {ticker} changed {change_30:.2%} in the last 30 days "
        f"and {change_7:.2%} in the last 7 days."
    )
    return desc


def query_hf_model(text):
    if not HF_API_TOKEN:
        raise RuntimeError("Ustaw zmienną środowiskową HF_API_TOKEN.")
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        return None

    if r.status_code != 200:
        print(f"API zwróciło {r.status_code}: {r.text[:200]}")
        return None

    try:
        return r.json()
    except Exception:
        print("Nie udało się sparsować JSON z odpowiedzi.")
        return None


def select_top_ai_stocks(tickers, n=10, log_path="ai_predictions.csv"):
    results = []
    all_predictions = []

    for t in tickers:
        text = summarize_stock_trend(t)
        if not text:
            continue

        pred = query_hf_model(text)
        if not pred:
            continue

        print(f"\n [API RESPONSE] {t} → {pred}\n")

        try:
            if isinstance(pred, list):
                inner = pred[0] if (len(pred) > 0 and isinstance(pred[0], list)) else pred
                valid_entries = [x for x in inner if isinstance(x, dict) and "label" in x]
                if not valid_entries:
                    continue
                entry = max(valid_entries, key=lambda x: x.get("score", 0.0))
                label = entry.get("label", "").lower()
                score = float(entry.get("score", 0.0))
            else:
                print(f"Nieoczekiwany format odpowiedzi dla {t}: {type(pred)}")
                continue
        except Exception as e:
            print(f"Błąd interpretacji predykcji dla {t}: {e}")
            continue

        all_predictions.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": t,
            "Label": label,
            "Confidence": score
        })

        if "positive" in label:
            results.append({"Ticker": t, "Label": label, "Confidence": score})

        time.sleep(0.4)

    if all_predictions:
        log_df = pd.DataFrame(all_predictions)
        if os.path.exists(log_path):
            old = pd.read_csv(log_path)
            log_df = pd.concat([old, log_df], ignore_index=True)
        log_df.to_csv(log_path, index=False)
        print(f"💾 Zapisano log predykcji: {log_path}")

    if not results:
        print("Model nie zwrócił żadnych pozytywnych nastrojów.")
        return pd.DataFrame(columns=["Ticker", "Label", "Confidence"])

    df = pd.DataFrame(results).sort_values("Confidence", ascending=False).head(n)
    print(f"Wybrano {len(df)} spółek z najwyższym pozytywnym trendem AI.")
    return df
