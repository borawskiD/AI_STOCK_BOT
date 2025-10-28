import os
import requests
import pandas as pd

CACHE_FILE = "nasdaq100_companies.csv"


def get_nasdaq100_tickers(cache_path: str = CACHE_FILE) -> pd.DataFrame:
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            if "Ticker" in df.columns and len(df) >= 80:
                print(f"💾 Wczytano {len(df)} spółek z cache.")
                return df
        except Exception as e:
            print(f"⚠️ Problem z wczytaniem cache: {e}")

    print("Pobieram dane z API NASDAQ...")
    url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Błąd połączenia z API NASDAQ: {e}")
        if os.path.exists(cache_path):
            print("💾 Używam starszego cache.")
            return pd.read_csv(cache_path)
        else:
            raise SystemExit("Brak danych NASDAQ i brak cache — przerwano działanie.")

    try:
        rows = payload["data"]["data"]["rows"]
    except (KeyError, TypeError):
        print("Nieoczekiwany format odpowiedzi API NASDAQ.")
        print("Dane otrzymane:", payload.keys())
        raise SystemExit("Nie można sparsować danych NASDAQ.")

    df = pd.DataFrame(rows)
    if not {"symbol", "companyName"}.issubset(df.columns):
        print("Brak wymaganych kolumn w danych NASDAQ.")
        raise SystemExit("API zwróciło niekompletne dane.")

    df = df[["symbol", "companyName"]].rename(columns={"symbol": "Ticker", "companyName": "Company"})

    df = df[df["Ticker"].notna()]
    df["Ticker"] = df["Ticker"].str.strip().str.upper()
    df["Company"] = df["Company"].str.strip()

    df.to_csv(cache_path, index=False)
    print(f"✅ Zapisano listę {len(df)} spółek do cache: {cache_path}")

    return df


if __name__ == "__main__":
    df = get_nasdaq100_tickers()
    print(df.head())