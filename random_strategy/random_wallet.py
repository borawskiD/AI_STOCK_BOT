import time
import random
import numpy as np
import pandas as pd
import yfinance as yf


def generate_random_portfolio(
    tickers,
    total_investment=10_000,
    allow_fractional=True,
    random_seed=None
):

    if random_seed is not None:
        np.random.seed(random_seed)

    n = np.random.randint(5, 21)
    selected = list(np.random.choice(tickers, n, replace=False))

    weights = np.random.random(n)
    weights /= weights.sum()
    allocations = total_investment * weights

    portfolio = pd.DataFrame({
        "Ticker": selected,
        "Weight": weights,
        "Investment($)": allocations
    })

    for attempt in range(3):
        try:
            data = yf.download(
                selected,
                period="1d",
                progress=False,
                auto_adjust=True,
                threads=True
            )

            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"].iloc[-1]
            else:
                prices = data["Close"].iloc[-1]
            break
        except Exception as e:
            time.sleep(random.uniform(1.0, 2.0))
    else:
        prices_dict = {}
        for t in selected:
            try:
                info = yf.Ticker(t).history(period="1d", auto_adjust=True)
                if not info.empty:
                    prices_dict[t] = info["Close"].iloc[-1]
            except Exception:
                continue
        prices = pd.Series(prices_dict)

    prices = prices.dropna()
    available_tickers = list(prices.index)
    missing = set(selected) - set(available_tickers)
    if missing:
        print(f"Brak danych dla: {', '.join(missing)}")
        portfolio = portfolio[portfolio["Ticker"].isin(available_tickers)]

    portfolio["Price"] = [prices[t] for t in portfolio["Ticker"]]

    if allow_fractional:
        portfolio["Shares"] = portfolio["Investment($)"] / portfolio["Price"]
        dust = 0.0
    else:
        portfolio["Shares"] = np.floor(portfolio["Investment($)"] / portfolio["Price"])
        portfolio["Investment($)"] = portfolio["Shares"] * portfolio["Price"]
        dust = total_investment - portfolio["Investment($)"].sum()

    portfolio["CurrentValue($)"] = portfolio["Shares"] * portfolio["Price"]

    portfolio = portfolio[["Ticker", "Weight", "Investment($)", "Price", "Shares", "CurrentValue($)"]]

    return portfolio.reset_index(drop=True), dust