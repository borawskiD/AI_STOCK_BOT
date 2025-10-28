import os
import time
import html
import random
import requests
import pandas as pd
from urllib.parse import quote_plus
from datetime import datetime

from data_fetcher import load_close_series

HF_MODEL_ID = "ProsusAI/finbert"
HF_ROUTER = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
RESULTS_DIR = "results"
NEWS_LOG_FILE = os.path.join(RESULTS_DIR, "ai_news_predictions.csv")


def _ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def _google_news_rss_url(query: str, days: int = 7, lang: str = "en-US", region: str = "US"):
    return (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(query)}%20when:{days}d"
        f"&hl={lang}&gl={region.split('-')[-1]}&ceid={region}:en"
    )


def fetch_news_for_ticker(ticker: str, days: int = 7, max_articles: int = 12):
    q = f"{ticker} stock"
    url = _google_news_rss_url(q, days=days)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[{ticker}] Błąd pobierania RSS: {e}")
        return []

    titles = []
    text = resp.text
    start = 0
    while True:
        i = text.find("<title>", start)
        if i == -1:
            break
        j = text.find("</title>", i + 7)
        if j == -1:
            break
        title = html.unescape(text[i + 7:j]).strip()
        titles.append(title)
        start = j + 8

    if titles:
        titles = titles[1:]

    uniq = []
    seen = set()
    for t in titles:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
        if len(uniq) >= max_articles:
            break
    return uniq


def _hf_inference_sentiment(texts, hf_token: str):
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }

    results = []
    BATCH = 8
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        payload = {"inputs": batch}
        try:
            r = requests.post(HF_ROUTER, headers=headers, json=payload, timeout=30)
            if r.status_code == 404:
                time.sleep(1.0)
                r = requests.post(HF_ROUTER, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            out = r.json()
            results.extend(out if isinstance(out, list) else [out])
        except Exception as e:
            for _ in batch:
                results.append([{"label": "neutral", "score": 1.0}])
        time.sleep(0.25)
    return results


def _score_from_probs(probs_list):
    p_pos = p_neg = 0.0
    for item in probs_list:
        lab = item.get("label", "").lower()
        sc = float(item.get("score", 0.0))
        if "pos" in lab:
            p_pos = max(p_pos, sc)
        elif "neg" in lab:
            p_neg = max(p_neg, sc)
    return p_pos - p_neg


def analyze_news_sentiment(tickers, days: int = 7, max_articles: int = 12, save_log: bool = True):
    _ensure_results_dir()

    total_articles = 0
    rows = []

    print(f"Start analizy newsów dla {len(tickers)} spółek (okres {days} dni)\n")

    for idx, t in enumerate(tickers, start=1):
        print(f"🔍 [{idx}/{len(tickers)}] Analiza: {t} ...", end=" ")

        titles = fetch_news_for_ticker(t, days=days, max_articles=max_articles)
        news_count = len(titles)
        total_articles += news_count

        if not titles:
            print("brak newsów.")
            last_series = load_close_series(t)
            last = float(last_series.iloc[-1]) if len(last_series) else float("nan")
            rows.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": t,
                "NewsCount": 0,
                "MeanSentiment": 0.0,
                "LastClose": last,
                "PredictedGrowth": 0.0
            })
            continue

        preds = _hf_inference_sentiment(titles, HF_API_TOKEN)

        per_article = []
        for p in preds:
            inner = p[0] if (isinstance(p, list) and len(p) > 0 and isinstance(p[0], list)) else p
            if isinstance(inner, list):
                score = _score_from_probs(inner)
            elif isinstance(inner, dict):
                score = _score_from_probs([inner])
            else:
                score = 0.0
            per_article.append(score)

        mean_sent = float(pd.Series(per_article).mean()) if per_article else 0.0

        predicted_growth = max(0.05, mean_sent * 1.5)

        last_series = load_close_series(t)
        last = float(last_series.iloc[-1]) if len(last_series) else float("nan")

        print(f"{news_count} newsów | sentyment={mean_sent:+.3f} | prognoza={predicted_growth:.3f}")

        rows.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": t,
            "NewsCount": news_count,
            "MeanSentiment": mean_sent,
            "LastClose": last,
            "PredictedGrowth": predicted_growth
        })

        time.sleep(0.15 + random.random() * 0.1)

    df = pd.DataFrame(rows)

    if save_log:
        df.to_csv(NEWS_LOG_FILE, index=False)
        print(f"\n Zapisano log analizy newsów: {NEWS_LOG_FILE}")

    positive = (df["MeanSentiment"] > 0).sum()
    neutral = (df["MeanSentiment"] == 0).sum()
    negative = (df["MeanSentiment"] < 0).sum()
    print(f"\n Podsumowanie sentymentów:")
    print(f"   • pozytywne: {positive}")
    print(f"   • neutralne: {neutral}")
    print(f"   • negatywne: {negative}")
    print(f"   • łączna liczba zebranych newsów: {total_articles}\n")

    df = df.sort_values(["PredictedGrowth", "NewsCount"], ascending=[False, False]).reset_index(drop=True)
    return df


def select_top_by_news(df_news: pd.DataFrame, n: int = 10):
    """Zwraca DataFrame TOP N spółek do portfela."""
    if df_news.empty:
        return pd.DataFrame(columns=["Ticker", "LastClose", "PredictedGrowth"])
    top = df_news.head(n)[["Ticker", "LastClose", "PredictedGrowth"]].copy()
    return top
