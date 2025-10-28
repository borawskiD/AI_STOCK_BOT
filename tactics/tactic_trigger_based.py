import os
import yfinance as yf
import pandas as pd
from datetime import datetime

def execute_trigger_based(strategy_func, tickers, total_investment, allow_fractional=True,
                          save_path=None, drop_threshold=0.05):

    print(f"Taktyka: TRIGGER-BASED (aktualizacja przy spadku > {drop_threshold*100:.1f}%)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        old_portfolio = pd.read_csv(save_path)
        tickers_old = old_portfolio["Ticker"].tolist()
        shares = old_portfolio["Shares"].tolist()

        try:
            data = yf.download(tickers_old, period="1d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"].iloc[-1]
            else:
                prices = data["Close"].iloc[-1]
            new_value = sum(prices[t] * s for t, s in zip(tickers_old, shares) if t in prices)
        except Exception:
            new_value = old_portfolio["CurrentValue($)"].sum()

        old_value = old_portfolio["CurrentValue($)"].sum()
        change = (new_value - old_value) / old_value

        print(f"Zmiana portfela: {change*100:.2f}%")

        if change > -drop_threshold:
            print("Brak potrzeby aktualizacji — spadek niewielki.")
            return old_portfolio, 0.0
        else:
            print("Wykryto spadek — przebudowuję portfel.")
    else:
        print("Tworzę nowy portfel (brak wcześniejszego).")

    portfolio, dust = strategy_func(
        tickers=tickers,
        total_investment=total_investment,
        allow_fractional=allow_fractional
    )
    portfolio["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    portfolio.to_csv(save_path, index=False)
    print(f"Portfel zaktualizowany: {save_path}")
    return portfolio, dust
