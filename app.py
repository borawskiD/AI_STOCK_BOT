import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="📊 AI Portfolio Dashboard", layout="wide")
st.title("AI Stock Strategy Dashboard")

RESULTS_DIR = "results"
PORTFOLIO_DIR = os.path.join(RESULTS_DIR, "portfolios")
DATA_DIR = "data"
HISTORY_FILE = os.path.join(RESULTS_DIR, "portfolio_history.csv")
COMPANIES_FILE = os.path.join(DATA_DIR, "nasdaq100_companies.csv")

if os.path.exists(COMPANIES_FILE):
    companies = pd.read_csv(COMPANIES_FILE)
else:
    st.warning("Nie znaleziono pliku `nasdaq100_companies.csv` w katalogu data/")
    companies = pd.DataFrame(columns=["Ticker", "Company"])

if os.path.exists(PORTFOLIO_DIR):
    portfolio_files = sorted([f for f in os.listdir(PORTFOLIO_DIR) if f.endswith(".csv")])
else:
    st.error("Brak katalogu results/portfolios — wygeneruj portfele przed uruchomieniem dashboardu.")
    st.stop()

def label_from_filename(fname):
    parts = fname.replace(".csv", "").split("_")
    strategy = parts[0].upper()
    tactic = parts[1].capitalize() if len(parts) > 1 else "Static"
    name_map = {
        "AI": "AI Trend",
        "NEWS": "News Sentiment",
        "RANDOM": "Random"
    }
    strat_label = name_map.get(strategy, strategy)
    return f"{strat_label} — {tactic}"

portfolio_labels = {label_from_filename(f): f for f in portfolio_files}

selected_label = st.sidebar.selectbox("Wybierz portfel:", list(portfolio_labels.keys()))
selected_file = portfolio_labels[selected_label]
path = os.path.join(PORTFOLIO_DIR, selected_file)

if os.path.exists(path):
    df = pd.read_csv(path)
    df = df.merge(companies, on="Ticker", how="left")

    df["Label"] = df.apply(
        lambda row: f"{row['Ticker']} — {row['Company'][:40]}..." if pd.notna(row["Company"]) else row["Ticker"],
        axis=1
    )

    st.subheader(f"{selected_label}")
    st.dataframe(df[["Ticker", "Company", "Shares", "Price", "CurrentValue($)"]])

    if "Weight" in df.columns:
        fig_pie = px.pie(
            df,
            names="Label",
            values="Weight",
            title="Udział spółek w portfelu"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    if "CurrentValue($)" in df.columns:
        fig_bar = px.bar(
            df,
            x="Label",
            y="CurrentValue($)",
            title="Aktualna wartość spółek",
            text_auto=".2s"
        )
        fig_bar.update_layout(xaxis_title=None, yaxis_title="Wartość ($)", xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

    total_val = df["CurrentValue($)"].sum()
    st.metric("Łączna wartość portfela", f"${total_val:,.2f}")
else:
    st.warning("Brak danych dla wybranego portfela.")

st.markdown("---")
st.subheader("Historia wartości portfeli (Time Series)")

if os.path.exists(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
    if "Value($)" not in hist.columns:
        st.error("Plik historii ma niepoprawny format (brak kolumny 'Value($)').")
    else:
        strategies = sorted(hist["Strategy"].unique())
        tactics = sorted(hist["Tactic"].unique())
        col1, col2 = st.columns(2)
        with col1:
            selected_strats = st.multiselect("Wybierz strategie:", strategies, default=strategies)
        with col2:
            selected_tacts = st.multiselect("Wybierz taktyki:", tactics, default=tactics)

        filtered = hist[(hist["Strategy"].isin(selected_strats)) & (hist["Tactic"].isin(selected_tacts))]

        fig_hist = px.line(
            filtered,
            x="Timestamp",
            y="Value($)",
            color="Portfolio",
            title="Zmiana wartości portfeli w czasie",
            markers=True
        )
        fig_hist.update_traces(mode="lines+markers")
        fig_hist.update_layout(xaxis_title="Czas", yaxis_title="Wartość ($)")
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_growth = px.line(
            filtered,
            x="Timestamp",
            y="Change(%)",
            color="Portfolio",
            title="Wzrost/Spadek procentowy portfeli w czasie",
            markers=True
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        last_vals = (
            filtered.sort_values("Timestamp")
            .groupby("Portfolio")
            .tail(1)
            .set_index("Portfolio")
        )
        st.subheader("Ostatnie wyniki")
        st.dataframe(
            last_vals[["Strategy", "Tactic", "Value($)", "Change(%)"]]
            .sort_values("Change(%)", ascending=False)
            .style.format({"Value($)": "${:,.2f}", "Change(%)": "{:+.2f}%"})
        )
else:
    st.info("Brak historii. Uruchom `portfolio_evaluator.py`, by zebrać dane.")
