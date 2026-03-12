"""
FinTech Fraud Detection — Interactive Dashboard
=================================================
Streamlit dashboard providing real-time visibility into fraud detection
metrics, transaction analytics, and ML model performance.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from db_config import get_engine

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinTech Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #0f3460;
    }
    .stMetric label { color: #8892b0 !important; font-size: 14px !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ccd6f6 !important; font-size: 28px !important; }
    h1 { color: #64ffda !important; }
    h2, h3 { color: #ccd6f6 !important; }
    .fraud-high { color: #ff6b6b; font-weight: bold; }
    .fraud-low { color: #51cf66; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    """Load analytics data from database."""
    engine = get_engine()
    try:
        df = pd.read_sql("SELECT * FROM transactions_analytics_ready", engine)
        df["transaction_ts"] = pd.to_datetime(df["transaction_ts"])
        return df
    except Exception as e:
        st.error(f"Database not populated yet. Run `python run_pipeline.py` first.\n\nError: {e}")
        return pd.DataFrame()


def main():
    # ── Header ──────────────────────────────────────────────────────
    st.title("🛡️ FinTech Fraud Detection Platform")
    st.markdown("*Real-time transaction monitoring & fraud analytics for UK financial operations*")

    df = load_data()
    if df.empty:
        st.warning("⚠️ No data found. Please run the pipeline first: `python run_pipeline.py`")
        st.stop()

    # ── Sidebar Filters ─────────────────────────────────────────────
    st.sidebar.header("🔍 Filters")

    if "merchant_category" in df.columns:
        categories = ["All"] + sorted(df["merchant_category"].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Merchant Category", categories)
        if selected_category != "All":
            df = df[df["merchant_category"] == selected_category]

    if "region" in df.columns:
        regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
        selected_region = st.sidebar.selectbox("Customer Region", regions)
        if selected_region != "All":
            df = df[df["region"] == selected_region]

    fraud_filter = st.sidebar.radio("Fraud Filter", ["All", "Fraud Only", "Legitimate Only"])
    if fraud_filter == "Fraud Only":
        df = df[df["is_fraud"] == 1]
    elif fraud_filter == "Legitimate Only":
        df = df[df["is_fraud"] == 0]

    # ── KPI Cards ───────────────────────────────────────────────────
    st.markdown("---")
    total_txns = len(df)
    total_volume = df["amount_gbp"].sum()
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = fraud_count / max(total_txns, 1) * 100
    fraud_exposure = df[df["is_fraud"] == 1]["amount_gbp"].sum()
    avg_txn = df["amount_gbp"].mean()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📊 Total Transactions", f"{total_txns:,}")
    col2.metric("💰 Total Volume (GBP)", f"£{total_volume:,.0f}")
    col3.metric("🚨 Fraud Count", f"{fraud_count:,}")
    col4.metric("📈 Fraud Rate", f"{fraud_rate:.2f}%")
    col5.metric("💸 Fraud Exposure", f"£{fraud_exposure:,.0f}")
    col6.metric("📉 Avg Transaction", f"£{avg_txn:.2f}")

    # ── Row 1: Fraud by Type + Daily Trend ──────────────────────────
    st.markdown("---")
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("🔴 Fraud by Type")
        fraud_df = df[df["is_fraud"] == 1]
        if not fraud_df.empty and "fraud_type" in fraud_df.columns:
            type_counts = fraud_df["fraud_type"].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = sns.color_palette("rocket", n_colors=len(type_counts))
            bars = ax.barh(type_counts.index, type_counts.values, color=colors)
            ax.set_xlabel("Count", color="#8892b0")
            ax.set_facecolor("#0E1117")
            fig.patch.set_facecolor("#0E1117")
            ax.tick_params(colors="#8892b0")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#233554")
            ax.spines["left"].set_color("#233554")
            for bar, val in zip(bars, type_counts.values):
                ax.text(val + 2, bar.get_y() + bar.get_height()/2, f"{val:,}",
                       va="center", color="#64ffda", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No fraud data to display")

    with right_col:
        st.subheader("📅 Daily Transaction Trend")
        daily = df.groupby(df["transaction_ts"].dt.date).agg(
            total=("transaction_id", "count"),
            fraud=("is_fraud", "sum"),
            volume=("amount_gbp", "sum"),
        ).reset_index()
        daily.columns = ["date", "total", "fraud", "volume"]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.fill_between(daily["date"], daily["total"], alpha=0.3, color="#64ffda")
        ax1.plot(daily["date"], daily["total"], color="#64ffda", linewidth=2, label="Total")
        ax1.set_ylabel("Total Transactions", color="#64ffda")
        ax1.tick_params(axis="y", labelcolor="#64ffda")

        ax2 = ax1.twinx()
        ax2.bar(daily["date"], daily["fraud"], alpha=0.7, color="#ff6b6b", label="Fraud", width=0.8)
        ax2.set_ylabel("Fraud Count", color="#ff6b6b")
        ax2.tick_params(axis="y", labelcolor="#ff6b6b")

        ax1.set_facecolor("#0E1117")
        fig.patch.set_facecolor("#0E1117")
        ax1.tick_params(colors="#8892b0")
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_color("#233554")
        ax2.spines["top"].set_visible(False)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 2: Category Heatmap + Risk Distribution ─────────────────
    st.markdown("---")
    left_col2, right_col2 = st.columns(2)

    with left_col2:
        st.subheader("🔥 Fraud Heatmap: Category × Hour")
        if "hour_of_day" in df.columns and "merchant_category" in df.columns:
            heatmap_data = df.pivot_table(
                values="is_fraud", index="merchant_category",
                columns="hour_of_day", aggfunc="mean"
            ).fillna(0) * 100

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False,
                       ax=ax, cbar_kws={"label": "Fraud Rate %"})
            ax.set_xlabel("Hour of Day", color="#8892b0")
            ax.set_ylabel("Category", color="#8892b0")
            ax.set_facecolor("#0E1117")
            fig.patch.set_facecolor("#0E1117")
            ax.tick_params(colors="#8892b0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with right_col2:
        st.subheader("📊 Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        legit = df[df["is_fraud"] == 0]["amount_gbp"].clip(upper=1000)
        fraud_amts = df[df["is_fraud"] == 1]["amount_gbp"].clip(upper=1000)
        ax.hist(legit, bins=50, alpha=0.6, color="#64ffda", label="Legitimate", density=True)
        if len(fraud_amts) > 0:
            ax.hist(fraud_amts, bins=50, alpha=0.6, color="#ff6b6b", label="Fraud", density=True)
        ax.set_xlabel("Amount (GBP, capped at £1,000)", color="#8892b0")
        ax.set_ylabel("Density", color="#8892b0")
        ax.legend(facecolor="#1a1a2e", edgecolor="#233554", labelcolor="#ccd6f6")
        ax.set_facecolor("#0E1117")
        fig.patch.set_facecolor("#0E1117")
        ax.tick_params(colors="#8892b0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#233554")
        ax.spines["left"].set_color("#233554")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 3: ML Model Performance ─────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 ML Model Performance")

    models_dir = PROJECT_ROOT / "ml" / "models"
    feat_img = models_dir / "feature_importance.png"
    conf_img = models_dir / "confusion_matrix.png"

    if feat_img.exists() or conf_img.exists():
        ml_col1, ml_col2 = st.columns(2)
        if feat_img.exists():
            with ml_col1:
                st.image(str(feat_img), caption="Top 15 Feature Importance", use_container_width=True)
        if conf_img.exists():
            with ml_col2:
                st.image(str(conf_img), caption="Confusion Matrix", use_container_width=True)
    else:
        st.info("ML model not trained yet. Run `python run_pipeline.py` to train and generate model artifacts.")

    # ── Row 4: Recent Fraud Alerts ──────────────────────────────────
    st.markdown("---")
    st.subheader("🚨 Recent Fraud Alerts")

    recent_fraud = df[df["is_fraud"] == 1].sort_values("transaction_ts", ascending=False).head(20)
    if not recent_fraud.empty:
        display_cols = [
            "transaction_id", "transaction_ts", "amount_gbp",
            "fraud_type", "merchant_category", "terminal_country", "region"
        ]
        display_cols = [c for c in display_cols if c in recent_fraud.columns]
        st.dataframe(
            recent_fraud[display_cols].style.format({"amount_gbp": "£{:.2f}"}),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No fraud transactions to display")

    # ── Footer ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8892b0;'>"
        "🛡️ FinTech Fraud Detection Platform • Built with Python, XGBoost, SQLAlchemy & Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
