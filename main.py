import os
import time

from generator import get_nasdaq100_tickers
from data_fetcher import fetch_price_history

from ai_history_prediction_strategy.ai_growth_selector import analyze_growth, select_top_n
from ai_news_prediction_strategy.ai_news_sentiment_strategy import analyze_news_sentiment, select_top_by_news
from portfolio_generator import generate_portfolio
from random_strategy.random_wallet import generate_random_portfolio

TOTAL_INVESTMENT = 10_000
ALLOW_FRACTIONAL = True
TOP_N = 10
RESULTS_DIR = "results"
PORTFOLIO_DIR = os.path.join(RESULTS_DIR, "portfolios")


def ensure_results_dir():
    os.makedirs(PORTFOLIO_DIR, exist_ok=True)


def build_ai_portfolio(tickers):
    df_predictions = analyze_growth(tickers, window=20)
    if df_predictions.empty:
        raise RuntimeError("Brak wyników analizy trendów.")
    top_tickers = select_top_n(df_predictions, n=TOP_N)
    top_df = df_predictions[df_predictions["Ticker"].isin(top_tickers)]
    portfolio, dust = generate_portfolio(top_df, TOTAL_INVESTMENT, ALLOW_FRACTIONAL)
    return portfolio, dust


def build_random_portfolio(tickers):
    portfolio, dust = generate_random_portfolio(
        tickers,
        total_investment=TOTAL_INVESTMENT,
        allow_fractional=ALLOW_FRACTIONAL,
        random_seed=int(time.time())
    )
    return portfolio, dust


def build_news_portfolio(tickers):
    df_news = analyze_news_sentiment(tickers=tickers, days=7, max_articles=20, save_log=True)
    top_news_df = select_top_by_news(df_news, n=TOP_N)
    if top_news_df.empty:
        raise RuntimeError("Brak kandydatów do portfela news.")
    portfolio, dust = generate_portfolio(top_news_df, TOTAL_INVESTMENT, ALLOW_FRACTIONAL)
    return portfolio, dust


def main():

    ensure_results_dir()

    companies = get_nasdaq100_tickers()
    tickers = companies["Ticker"].tolist()
    fetch_price_history(tickers, period="6mo")

    strategies = {
        "AI": build_ai_portfolio,
        "NEWS": build_news_portfolio,
        "RANDOM": build_random_portfolio,
    }

    tactics = ["STATIC", "REGULAR", "TRIGGER"]

    for strat_name, build_fn in strategies.items():
        for tactic in tactics:
            print("\n" + "=" * 80)
            print(f"Generuję portfel: Strategia={strat_name}, Taktyka={tactic}")
            print("=" * 80)

            try:
                portfolio, dust = build_fn(tickers)
            except Exception as e:
                print(f"Błąd generowania portfela {strat_name}_{tactic}: {e}")
                continue

            # Zapis portfela do results/portfolios
            csv_name = f"{strat_name}_{tactic}.csv"
            path = os.path.join(PORTFOLIO_DIR, csv_name)
            portfolio.to_csv(path, index=False)

            total_value = portfolio["CurrentValue($)"].sum()
            print(f"Zapisano {path}")
            print(f"Wartość: ${total_value:,.2f}, Dust: ${dust:,.2f}")

    print("\nWszystkie 9 portfeli zostały wygenerowane i zapisane.")
    print(f"Lokalizacja: {PORTFOLIO_DIR}/")


if __name__ == "__main__":
    main()
