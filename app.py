import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="AI Portfolio Dashboard", layout="wide")

RESULTS_DIR = "results"
DATA_DIR = "data"
HISTORY_FILE = os.path.join(RESULTS_DIR, "portfolio_history.csv")
COMPANIES_FILE = os.path.join(DATA_DIR, "nasdaq100_companies.csv")

st.title("📈 AI Stock Strategy Dashboard")

if os.path.exists(COMPANIES_FILE):
    companies = pd.read_csv(COMPANIES_FILE)
else:
    st.warning("Nie znaleziono pliku z nazwami spółek (nasdaq100_companies.csv)")
    companies = pd.DataFrame(columns=["Ticker", "Company"])

strategies = {
    "AI Trend": "portfolio_ai.csv",
    "Random": "portfolio_random.csv",
    "News Sentiment": "portfolio_news.csv"
}

selected = st.sidebar.selectbox("Wybierz strategię:", list(strategies.keys()))
path = os.path.join(RESULTS_DIR, strategies[selected])

if os.path.exists(path):
    df = pd.read_csv(path)
    df = df.merge(companies, on="Ticker", how="left")

    df["Label"] = df.apply(
        lambda row: f"{row['Ticker']} — {row['Company'][:35]}..." if pd.notna(row["Company"]) else row["Ticker"],
        axis=1
    )

    st.subheader(f"{selected} Portfolio")
    st.dataframe(df[["Ticker", "Company", "Shares", "Price", "CurrentValue($)"]])

    if "Weight" in df.columns:
        fig = px.pie(
            df,
            names="Label",
            values="Weight",
            title="Udział spółek w portfelu"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "CurrentValue($)" in df.columns:
        fig2 = px.bar(
            df,
            x="Label",
            y="CurrentValue($)",
            title="Aktualna wartość spółek",
            text_auto=".2s"
        )
        fig2.update_layout(xaxis_title=None, yaxis_title="Wartość ($)", xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Brak danych dla wybranej strategii.")

# --- Sekcja historii ---
st.markdown("---")
st.subheader("📉 Historia wartości portfeli (Time Series)")

if os.path.exists(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
    fig3 = px.line(
        hist,
        x="Timestamp",
        y="Value",
        color="Strategy",
        title="Wartość portfeli w czasie",
        markers=True
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Brak historii. Uruchom `update_portfolio_history.py`, by zebrać dane.")
