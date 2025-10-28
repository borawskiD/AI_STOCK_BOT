# ai_news_sentiment_strategy_optimized.py
import os
import time
import html
import random
import requests
import pandas as pd
from urllib.parse import quote_plus
from datetime import datetime
from dotenv import load_dotenv

from data_fetcher import load_close_series

# --- KONFIGURACJA ---
load_dotenv()
HF_MODEL_ID = "ProsusAI/finbert"
HF_ROUTER = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

RESULTS_DIR = "results"
NEWS_LOG_FILE = os.path.join(RESULTS_DIR, "ai_news_predictions.csv")
CACHE_FILE = os.path.join(RESULTS_DIR, "news_cache.csv")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise EnvironmentError("❌ Brak HF_API_TOKEN. Ustaw w .env lub GitHub Secrets.")

print(f"🔐 Token HF załadowany ({HF_API_TOKEN[:10]}...)\n")

# --- UTYLITY ---

def _ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def _google_news_rss_url(query: str, days: int = 7, lang: str = "en-US", region: str = "US"):
    """Buduje URL do Google News RSS."""
    return (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(query)}%20when:{days}d"
        f"&hl={lang}&gl={region.split('-')[-1]}&ceid={region}:en"
    )


def _load_news_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=["Ticker", "Title", "Sentiment", "Timestamp"])


def _save_news_cache(df_cache):
    df_cache.to_csv(CACHE_FILE, index=False)


def fetch_news_for_ticker(ticker: str, days: int = 7, max_articles: int = 12):
    """Pobiera nagłówki newsów z Google News RSS."""
    q = f"{ticker} stock"
    url = _google_news_rss_url(q, days=days)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
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
        titles = titles[1:]  # pomiń tytuł feedu

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
    """Batchowa inferencja sentymentu (8 newsów na zapytanie)."""
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
            r = requests.post(HF_ROUTER, headers=headers, json=payload, timeout=40)
            if r.status_code != 200:
                print(f"⚠️ [HF] {r.status_code}: {r.text[:120]}")
                # fallback neutral
                for _ in batch:
                    results.append([{"label": "neutral", "score": 1.0}])
                continue
            out = r.json()
            results.extend(out if isinstance(out, list) else [out])
        except Exception as e:
            print(f"⚠️ [HF Error] {e}")
            for _ in batch:
                results.append([{"label": "neutral", "score": 1.0}])
        time.sleep(0.25)  # backoff między batchami
    return results


def _score_from_probs(probs_list):
    """Liczy prosty sentyment z wyników FinBERT."""
    p_pos = p_neg = 0.0
    for item in probs_list:
        lab = item.get("label", "").lower()
        sc = float(item.get("score", 0.0))
        if "pos" in lab:
            p_pos = max(p_pos, sc)
        elif "neg" in lab:
            p_neg = max(p_neg, sc)
    return p_pos - p_neg


def analyze_news_sentiment(tickers, days: int = 7, max_articles: int = 12, save_log: bool = True, max_total_news: int = 1100):
    """Analiza sentymentu newsów dla listy tickerów (z cacheowaniem)."""
    _ensure_results_dir()
    cache_df = _load_news_cache()

    rows = []
    total_articles = 0
    print(f"🚀 Start analizy newsów ({len(tickers)} spółek, {days} dni)\n")

    for idx, t in enumerate(tickers, start=1):
        print(f"🔍 [{idx}/{len(tickers)}] {t} ...", end=" ")

        titles = fetch_news_for_ticker(t, days=days, max_articles=max_articles)
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

        cached_titles = set(cache_df.loc[cache_df["Ticker"] == t, "Title"].str.lower().tolist())
        new_titles = [x for x in titles if x.lower() not in cached_titles]

        # jeśli wszystko w cache
        if not new_titles:
            cached_sent = cache_df.loc[cache_df["Ticker"] == t, "Sentiment"]
            mean_sent = float(cached_sent.mean()) if not cached_sent.empty else 0.0
            predicted_growth = max(0.05, mean_sent * 1.5)
            last_series = load_close_series(t)
            last = float(last_series.iloc[-1]) if len(last_series) else float("nan")
            print(f"{len(titles)} newsów (cache) | sentyment={mean_sent:+.3f} | prognoza={predicted_growth:.3f}")
            rows.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": t,
                "NewsCount": len(titles),
                "MeanSentiment": mean_sent,
                "LastClose": last,
                "PredictedGrowth": predicted_growth
            })
            continue

        # analiza nowych tytułów
        preds = _hf_inference_sentiment(new_titles, HF_API_TOKEN)
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

        # zapis do cache
        for title, score in zip(new_titles, per_article):
            cache_df.loc[len(cache_df)] = {
                "Ticker": t,
                "Title": title,
                "Sentiment": score,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        mean_sent = float(pd.Series(per_article).mean()) if per_article else 0.0
        predicted_growth = max(0.05, mean_sent * 1.5)
        last_series = load_close_series(t)
        last = float(last_series.iloc[-1]) if len(last_series) else float("nan")

        print(f"{len(titles)} newsów ({len(new_titles)} nowych) | sentyment={mean_sent:+.3f} | prognoza={predicted_growth:.3f}")

        rows.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": t,
            "NewsCount": len(titles),
            "MeanSentiment": mean_sent,
            "LastClose": last,
            "PredictedGrowth": predicted_growth
        })

        total_articles += len(new_titles)
        if total_articles > max_total_news:
            print("⚠️ Osiągnięto limit analizy newsów – przerywam, by oszczędzić tokeny.")
            break

        time.sleep(0.15 + random.random() * 0.1)

    _save_news_cache(cache_df)
    df = pd.DataFrame(rows)

    if save_log:
        df.to_csv(NEWS_LOG_FILE, index=False)
        print(f"\n💾 Zapisano log analizy: {NEWS_LOG_FILE}")

    print(f"\n📊 Podsumowanie: {total_articles} nowych newsów przetworzonych, {len(cache_df)} w cache.\n")

    df = df.sort_values(["PredictedGrowth", "NewsCount"], ascending=[False, False]).reset_index(drop=True)
    return df


def select_top_by_news(df_news: pd.DataFrame, n: int = 10):
    """Zwraca TOP N spółek wg PredictedGrowth."""
    if df_news.empty:
        return pd.DataFrame(columns=["Ticker", "LastClose", "PredictedGrowth"])
    return df_news.head(n)[["Ticker", "LastClose", "PredictedGrowth"]].copy()
