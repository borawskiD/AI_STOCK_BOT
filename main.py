import os
import time

from ai_news_prediction_strategy.ai_news_sentiment_strategy import analyze_news_sentiment, select_top_by_news
from generator import get_nasdaq100_tickers
from data_fetcher import fetch_price_history
from ai_history_prediction_strategy.ai_growth_selector import analyze_growth, select_top_n
from portfolio_generator import generate_portfolio
from random_strategy.random_wallet import generate_random_portfolio

TOTAL_INVESTMENT = 10_000
ALLOW_FRACTIONAL = True
TOP_N = 10
RESULTS_DIR = "results"


def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Utworzono katalog: {RESULTS_DIR}/")
    else:
        print(f"Katalog wyników istnieje: {RESULTS_DIR}/")

def main():
    companies = get_nasdaq100_tickers()
    tickers = companies["Ticker"].tolist()

    fetch_price_history(tickers, period="6mo")

    df_predictions = analyze_growth(tickers, window=20)
    if df_predictions.empty:
        print("Brak wyników analizy trendów.")
        return

    print("Najlepsze prognozy (TOP 10):")
    print(df_predictions.head(10))

    # 4️⃣ 🧠 Wybór najlepszych spółek AI
    top_tickers = select_top_n(df_predictions, n=TOP_N)
    print(f"Wybrane spółki (AI): {top_tickers}")

    if not top_tickers:
        print("Brak spółek do stworzenia portfela AI.")
        return

    # Wyciąg dane tylko dla wybranych spółek
    top_df = df_predictions[df_predictions["Ticker"].isin(top_tickers)]

    # 5️⃣ 💹 Stworzenie portfela AI
    print("\nGenerowanie portfela AI...")
    portfolio_ai, dust_ai = generate_portfolio(
        df_predictions=top_df,
        total_investment=TOTAL_INVESTMENT,
        allow_fractional=ALLOW_FRACTIONAL
    )

    ai_path = os.path.join(RESULTS_DIR, "portfolio_ai.csv")
    portfolio_ai.to_csv(ai_path, index=False)

    print("\nPortfel AI (prognoza wzrostu):")
    print(portfolio_ai)
    print(f"\nŁączna wartość portfela AI: ${portfolio_ai['CurrentValue($)'].sum():,.2f}")
    if dust_ai > 0:
        print(f"🪙 Niewykorzystane środki (dust): ${dust_ai:,.2f}")
    else:
        print("Brak niewykorzystanych środków w portfelu AI.")

    time.sleep(1)
    print("\n" + "="*70)
    print("Generowanie portfela losowego")
    print("="*70)

    portfolio_random, dust_random = generate_random_portfolio(
        tickers,
        total_investment=TOTAL_INVESTMENT,
        allow_fractional=ALLOW_FRACTIONAL,
        random_seed=int(time.time())
    )

    random_path = os.path.join(RESULTS_DIR, "portfolio_random.csv")
    portfolio_random.to_csv(random_path, index=False)

    print("\nPortfel losowy:")
    print(portfolio_random)
    print(f"\nŁączna wartość portfela losowego: ${portfolio_random['CurrentValue($)'].sum():,.2f}")
    if dust_random > 0:
        print(f"Niewykorzystane środki (dust): ${dust_random:,.2f}")
    else:
        print("Brak niewykorzystanych środków w portfelu losowym.")


   # 7️⃣ 📰 Strategia NEWS Sentiment
    print("\n" + "="*70)
    print("Generowanie portfela na podstawie newsów (sentyment AI)")
    print("="*70)

    df_news = analyze_news_sentiment(
        tickers=tickers,
        days=7,
        max_articles=20,
        save_log=True
    )

    top_news_df = select_top_by_news(df_news, n=TOP_N)
    if top_news_df.empty:
        print("Brak kandydatów do portfela news.")
    else:
        print("\nTOP z newsów:")
        print(top_news_df)

        portfolio_news, dust_news = generate_portfolio(
            df_predictions=top_news_df,
            total_investment=TOTAL_INVESTMENT,
            allow_fractional=ALLOW_FRACTIONAL
        )

        news_path = os.path.join("results", "portfolio_news.csv")
        portfolio_news.to_csv(news_path, index=False)
        print(f"Zapisano portfel NEWS: {news_path}")

        print("\nPortfel NEWS:")
        print(portfolio_news)
        print(f"\nŁączna wartość portfela NEWS: ${portfolio_news['CurrentValue($)'].sum():,.2f}")
        if dust_news > 0:
            print(f"Niewykorzystane środki (news dust): ${dust_news:,.2f}")
        else:
            print("Brak niewykorzystanych środków w portfelu NEWS.")


if __name__ == "__main__":
    main()
