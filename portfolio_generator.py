import numpy as np
import pandas as pd
import yfinance as yf


def generate_portfolio(df_predictions, total_investment=10_000, allow_fractional=True):
    df = df_predictions.copy()
    required_cols = {"Ticker", "PredictedGrowth", "LastClose"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Brak wymaganych kolumn: {required_cols - set(df.columns)}")

    df = df.dropna(subset=["PredictedGrowth", "LastClose"])
    if df.empty:
        raise ValueError("Brak danych predykcji do stworzenia portfela.")

    df = (
        df.groupby("Ticker", as_index=False)
        .agg({"PredictedGrowth": "mean", "LastClose": "last"})
    )

    df = df[df["PredictedGrowth"] > 0].copy()
    if df.empty:
        raise ValueError("Brak spółek z dodatnim przewidywanym wzrostem.")

    df["Weight"] = df["PredictedGrowth"] / df["PredictedGrowth"].sum()
    df["Investment($)"] = df["Weight"] * total_investment

    tickers = df["Ticker"].tolist()
    try:
        prices = yf.download(tickers, period="1d", progress=False, auto_adjust=True)["Close"].iloc[-1]
    except Exception:
        prices = pd.Series(dtype=float)

    df["Price"] = [prices.get(t, df.loc[df["Ticker"] == t, "LastClose"].iloc[0]) for t in tickers]

    if allow_fractional:
        df["Shares"] = df["Investment($)"] / df["Price"]
        df["Spent($)"] = df["Investment($)"]
        dust = 0.0
    else:
        df["Shares"] = np.floor(df["Investment($)"] / df["Price"])
        df["Spent($)"] = df["Shares"] * df["Price"]
        dust = float(total_investment - df["Spent($)"].sum())

    df["CurrentValue($)"] = df["Shares"] * df["Price"]

    df = (
        df.groupby("Ticker", as_index=False)
        .agg({
            "Weight": "sum",
            "Investment($)": "sum",
            "Price": "mean",
            "Shares": "sum",
            "CurrentValue($)": "sum"
        })
    )

    df["Weight"] = df["Weight"].round(4)
    df["Investment($)"] = df["Investment($)"].round(2)
    df["Price"] = df["Price"].round(2)
    df["Shares"] = df["Shares"].round(4 if allow_fractional else 0)
    df["CurrentValue($)"] = df["CurrentValue($)"].round(2)

    total_value = df["CurrentValue($)"].sum()
    print(f"\n💵 Łączna wartość portfela: ${total_value:,.2f} (dust: ${dust:,.2f})")

    return df, dust