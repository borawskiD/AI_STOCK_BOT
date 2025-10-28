import os
import pandas as pd
from datetime import datetime

def execute_regular(strategy_func, tickers, total_investment, allow_fractional=True,
                    save_path=None, update_interval_days=2):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        last_update = datetime.fromtimestamp(os.path.getmtime(save_path))
        delta_days = (datetime.now() - last_update).days
        if delta_days < update_interval_days:
            print(f"Aktualizacja pominięta — minęło tylko {delta_days} dni.")
            return pd.read_csv(save_path), 0.0

    portfolio, dust = strategy_func(
        tickers=tickers,
        total_investment=total_investment,
        allow_fractional=allow_fractional
    )

    portfolio["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    portfolio.to_csv(save_path, index=False)
    print(f"Portfel zaktualizowany: {save_path}")

    return portfolio, dust
