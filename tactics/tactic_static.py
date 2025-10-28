import os

def execute_static(strategy_func, tickers, total_investment, allow_fractional=True, save_path=None):
    print("Taktyka: STATIC (Kup i trzymaj)")

    portfolio, dust = strategy_func(
        tickers=tickers,
        total_investment=total_investment,
        allow_fractional=allow_fractional
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        portfolio.to_csv(save_path, index=False)
        print(f"Zapisano portfel STATIC → {save_path}")

    return portfolio, dust
