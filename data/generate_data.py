"""
Synthetic Financial Data Generator
===================================
Generates realistic UK financial transaction data with injected fraud patterns
for the FinTech Fraud Detection & Transaction Monitoring Platform.

Produces:
  - 50,000 customers (UK regions)
  - 5,000 merchants (8 categories with varying fraud base-rates)
  - 1,000,000 base transactions
  - ~2,300 injected fraud transactions across 4 fraud typologies

Final fraud rate target: 2–4% of total transactions.
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from faker import Faker
from sqlalchemy import create_engine

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

fake = Faker("en_GB")
Faker.seed(42)
np.random.seed(42)

UK_REGIONS = [
    "London", "Manchester", "Birmingham", "Leeds",
    "Glasgow", "Liverpool", "Bristol", "Sheffield",
]

MERCHANT_CATEGORIES = {
    "Retail": 0.01,
    "Travel": 0.03,
    "Restaurants": 0.01,
    "Crypto Exchange": 0.12,
    "Online Gaming": 0.08,
    "Electronics": 0.05,
    "ATM Withdrawal": 0.04,
    "Money Transfer": 0.09,
}

CURRENCIES = ["GBP", "USD", "EUR", "JPY", "AUD", "CAD", "CHF"]
FX_TO_GBP = {
    "GBP": 1.0, "USD": 0.79, "EUR": 0.86, "JPY": 0.0052,
    "AUD": 0.51, "CAD": 0.58, "CHF": 0.89,
}

TERMINAL_TYPES = ["pos", "online", "atm", "contactless"]
TERMINAL_COUNTRIES = ["GB", "US", "DE", "FR", "NL", "IE", "ES"]


def _get_engine():
    """Build a SQLAlchemy engine using the centralized db_config."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from db_config import get_engine, init_sqlite_schema
    engine = get_engine()
    if engine.dialect.name == "sqlite":
        init_sqlite_schema(engine)
    return engine


# ===================================================================
# PHASE 1A — Generate base entities
# ===================================================================

def generate_customers(n: int = 50_000) -> pd.DataFrame:
    """
    Generate synthetic UK bank customers.

    Business context: creates a realistic customer base spread across
    8 major UK cities with beta-distributed risk scores (most customers
    are low risk, matching real-world portfolios).
    """
    logger.info("Generating %s customers …", f"{n:,}")
    records = []
    for _ in range(n):
        records.append({
            "customer_id": str(uuid.uuid4()),
            "full_name": fake.name(),
            "email": fake.unique.email(),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=85),
            "account_opened": fake.date_between(start_date="-10y", end_date="today"),
            "region": np.random.choice(UK_REGIONS),
            "risk_score": round(np.random.beta(2, 8) * 10, 2),
        })
    df = pd.DataFrame(records)
    logger.info("  ✓ Customers generated: %s rows", f"{len(df):,}")
    return df


def generate_merchants(n: int = 5_000) -> pd.DataFrame:
    """
    Generate synthetic merchants across high-risk and standard categories.

    Business context: merchant mix reflects realistic category distribution
    with certain categories (Crypto Exchange, Money Transfer) flagged as
    high risk based on industry fraud base-rates.
    """
    logger.info("Generating %s merchants …", f"{n:,}")
    categories = list(MERCHANT_CATEGORIES.keys())
    high_risk_cats = {"Crypto Exchange", "Online Gaming", "Money Transfer"}
    records = []
    for _ in range(n):
        cat = np.random.choice(categories)
        records.append({
            "merchant_id": str(uuid.uuid4()),
            "merchant_name": fake.company(),
            "category": cat,
            "country_code": np.random.choice(
                TERMINAL_COUNTRIES, p=[0.55, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05]
            ),
            "is_high_risk": cat in high_risk_cats,
        })
    df = pd.DataFrame(records)
    logger.info("  ✓ Merchants generated: %s rows", f"{len(df):,}")
    return df


def generate_transactions(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n: int = 1_000_000,
) -> pd.DataFrame:
    """
    Generate 1M realistic base transactions with varied distributions.

    Business context: amount distributions are log-normal (most transactions
    small, few very large), timestamp spread across a full year with
    realistic intra-day patterns, and currency mix weighted toward GBP.
    """
    logger.info("Generating %s base transactions …", f"{n:,}")

    customer_ids = customers["customer_id"].values
    merchant_ids = merchants["merchant_id"].values
    merchant_cats = merchants.set_index("merchant_id")["category"].to_dict()
    merchant_countries = merchants.set_index("merchant_id")["country_code"].to_dict()

    # Pre-compute random arrays for speed
    cust_choices = np.random.choice(customer_ids, size=n)
    merch_choices = np.random.choice(merchant_ids, size=n)

    # Log-normal amounts (median ~£45, long tail)
    amounts = np.round(np.random.lognormal(mean=3.8, sigma=1.2, size=n), 2)
    amounts = np.clip(amounts, 0.50, 50_000.0)

    # Currencies weighted toward GBP
    currencies = np.random.choice(
        CURRENCIES,
        size=n,
        p=[0.60, 0.15, 0.12, 0.03, 0.04, 0.03, 0.03],
    )

    # Timestamps spread over past 365 days
    base_ts = datetime.utcnow() - timedelta(days=365)
    random_seconds = np.random.randint(0, 365 * 24 * 3600, size=n)
    timestamps = [base_ts + timedelta(seconds=int(s)) for s in random_seconds]

    terminal_types = np.random.choice(TERMINAL_TYPES, size=n, p=[0.35, 0.30, 0.15, 0.20])
    terminal_countries = np.random.choice(
        TERMINAL_COUNTRIES, size=n,
        p=[0.60, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
    )

    # Base fraud from merchant category base rates
    is_fraud_base = np.array([
        np.random.random() < MERCHANT_CATEGORIES.get(merchant_cats.get(m, "Retail"), 0.01)
        for m in merch_choices
    ])

    original_amounts = amounts.copy()
    gbp_amounts = np.array([
        round(amt * FX_TO_GBP.get(cur, 1.0), 2)
        for amt, cur in zip(original_amounts, currencies)
    ])

    # Decline and dispute flags (loosely correlated with fraud)
    is_declined = (np.random.random(n) < 0.03) | (is_fraud_base & (np.random.random(n) < 0.15))
    is_disputed = (np.random.random(n) < 0.01) | (is_fraud_base & (np.random.random(n) < 0.10))

    df = pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
        "customer_id": cust_choices,
        "merchant_id": merch_choices,
        "amount_gbp": gbp_amounts,
        "original_currency": currencies,
        "original_amount": original_amounts,
        "transaction_ts": timestamps,
        "terminal_type": terminal_types,
        "terminal_country": terminal_countries,
        "is_declined": is_declined,
        "is_disputed": is_disputed,
        "is_fraud": is_fraud_base,
        "fraud_type": np.where(is_fraud_base, "base_category_fraud", None),
    })

    logger.info("  ✓ Base transactions generated: %s rows", f"{len(df):,}")
    return df


# ===================================================================
# PHASE 1B — Inject specific fraud patterns
# ===================================================================

def inject_geographic_jumps(df: pd.DataFrame, count: int = 500) -> pd.DataFrame:
    """
    Inject geographic jump fraud: same card used in two countries within
    10–25 minutes (impossible travel).

    Business context: detects stolen card details being used simultaneously
    across geographic regions that are physically impossible to traverse.
    """
    logger.info("Injecting %s geographic_jump fraud pairs …", count)
    customer_ids = df["customer_id"].unique()
    fraud_rows = []
    for _ in range(count):
        cust = np.random.choice(customer_ids)
        ts1 = fake.date_time_between(start_date="-300d", end_date="-10d")
        gap_minutes = np.random.randint(10, 26)
        ts2 = ts1 + timedelta(minutes=gap_minutes)
        base = {
            "customer_id": cust,
            "merchant_id": np.random.choice(df["merchant_id"].unique()),
            "original_currency": "GBP",
            "terminal_type": "pos",
            "is_declined": False,
            "is_disputed": False,
            "is_fraud": True,
            "fraud_type": "geographic_jump",
        }
        amt1 = round(np.random.uniform(20, 500), 2)
        amt2 = round(np.random.uniform(20, 500), 2)
        fraud_rows.append({
            **base,
            "transaction_id": str(uuid.uuid4()),
            "amount_gbp": amt1, "original_amount": amt1,
            "transaction_ts": ts1,
            "terminal_country": "GB",
        })
        fraud_rows.append({
            **base,
            "transaction_id": str(uuid.uuid4()),
            "amount_gbp": amt2, "original_amount": amt2,
            "transaction_ts": ts2,
            "terminal_country": np.random.choice(["FR", "DE", "NL", "ES"]),
        })
    fraud_df = pd.DataFrame(fraud_rows)
    logger.info("  ✓ Geographic jump rows: %s", f"{len(fraud_df):,}")
    return pd.concat([df, fraud_df], ignore_index=True)


def inject_velocity_bursts(df: pd.DataFrame, count: int = 800) -> pd.DataFrame:
    """
    Inject velocity burst fraud: 5–9 rapid transactions from same card
    within a 60-minute window across multiple countries.

    Business context: compromised cards are often monetised quickly
    through multiple rapid purchases before the cardholder notices.
    """
    logger.info("Injecting %s velocity_burst fraud clusters …", count)
    customer_ids = df["customer_id"].unique()
    fraud_rows = []
    for _ in range(count):
        cust = np.random.choice(customer_ids)
        n_txns = np.random.randint(5, 10)
        ts_start = fake.date_time_between(start_date="-300d", end_date="-10d")
        for j in range(n_txns):
            amt = round(np.random.uniform(200, 3000), 2)
            fraud_rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust,
                "merchant_id": np.random.choice(df["merchant_id"].unique()),
                "amount_gbp": amt,
                "original_currency": "GBP",
                "original_amount": amt,
                "transaction_ts": ts_start + timedelta(minutes=np.random.randint(0, 60)),
                "terminal_type": np.random.choice(["pos", "online", "contactless"]),
                "terminal_country": np.random.choice(["GB", "US", "DE", "NL"]),
                "is_declined": False,
                "is_disputed": False,
                "is_fraud": True,
                "fraud_type": "velocity_burst",
            })
    fraud_df = pd.DataFrame(fraud_rows)
    logger.info("  ✓ Velocity burst rows: %s", f"{len(fraud_df):,}")
    return pd.concat([df, fraud_df], ignore_index=True)


def inject_account_takeover(
    df: pd.DataFrame, customers: pd.DataFrame, count: int = 400
) -> pd.DataFrame:
    """
    Inject account takeover fraud: high-value foreign transaction on a
    newly-opened account (< 30 days old).

    Business context: fraudsters open or hijack new accounts and immediately
    attempt high-value purchases before controls are established.
    """
    logger.info("Injecting %s account_takeover fraud transactions …", count)
    recent_cutoff = datetime.utcnow().date() - timedelta(days=30)
    new_accounts = customers[customers["account_opened"] >= recent_cutoff]
    if len(new_accounts) < count:
        new_accounts = customers.sample(count, random_state=42)
    chosen = new_accounts.sample(min(count, len(new_accounts)), random_state=42)
    fraud_rows = []
    for _, row in chosen.iterrows():
        amt = round(np.random.uniform(5000, 15000), 2)
        fraud_rows.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": row["customer_id"],
            "merchant_id": np.random.choice(df["merchant_id"].unique()),
            "amount_gbp": amt,
            "original_currency": np.random.choice(["USD", "EUR"]),
            "original_amount": amt,
            "transaction_ts": fake.date_time_between(start_date="-25d", end_date="-1d"),
            "terminal_type": "online",
            "terminal_country": np.random.choice(["US", "DE", "FR", "NL"]),
            "is_declined": False,
            "is_disputed": False,
            "is_fraud": True,
            "fraud_type": "account_takeover",
        })
    fraud_df = pd.DataFrame(fraud_rows)
    logger.info("  ✓ Account takeover rows: %s", f"{len(fraud_df):,}")
    return pd.concat([df, fraud_df], ignore_index=True)


def inject_card_testing(df: pd.DataFrame, count: int = 600) -> pd.DataFrame:
    """
    Inject card-testing fraud: 10–20 micro-transactions (£0.01–£2.00)
    within 5 minutes, followed by one large purchase (£500–£2,000).

    Business context: criminals test stolen card details with tiny charges;
    if those succeed, they immediately make a large purchase.
    """
    logger.info("Injecting %s card_testing fraud clusters …", count)
    customer_ids = df["customer_id"].unique()
    fraud_rows = []
    for _ in range(count):
        cust = np.random.choice(customer_ids)
        n_micro = np.random.randint(10, 21)
        ts_start = fake.date_time_between(start_date="-300d", end_date="-10d")
        for j in range(n_micro):
            amt = round(np.random.uniform(0.01, 2.00), 2)
            fraud_rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust,
                "merchant_id": np.random.choice(df["merchant_id"].unique()),
                "amount_gbp": amt,
                "original_currency": "GBP",
                "original_amount": amt,
                "transaction_ts": ts_start + timedelta(seconds=np.random.randint(0, 300)),
                "terminal_type": "online",
                "terminal_country": "GB",
                "is_declined": False,
                "is_disputed": False,
                "is_fraud": True,
                "fraud_type": "card_testing",
            })
        # Large follow-up transaction
        big_amt = round(np.random.uniform(500, 2000), 2)
        fraud_rows.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cust,
            "merchant_id": np.random.choice(df["merchant_id"].unique()),
            "amount_gbp": big_amt,
            "original_currency": "GBP",
            "original_amount": big_amt,
            "transaction_ts": ts_start + timedelta(minutes=6),
            "terminal_type": "online",
            "terminal_country": "GB",
            "is_declined": False,
            "is_disputed": False,
            "is_fraud": True,
            "fraud_type": "card_testing",
        })
    fraud_df = pd.DataFrame(fraud_rows)
    logger.info("  ✓ Card testing rows: %s", f"{len(fraud_df):,}")
    return pd.concat([df, fraud_df], ignore_index=True)


# ===================================================================
# PHASE 1C — Load to PostgreSQL
# ===================================================================

def load_to_postgres(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    transactions: pd.DataFrame,
    engine,
    batch_size: int = 10_000,
) -> None:
    """
    Bulk-load generated data into PostgreSQL in batches.

    Business context: staged loading ensures the database is not overwhelmed
    and allows progress tracking during large data ingestion jobs.
    """
    for table_name, df in [
        ("customers", customers),
        ("merchants", merchants),
        ("transactions", transactions),
    ]:
        logger.info("Loading %s → %s rows …", table_name, f"{len(df):,}")
        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            chunksize=batch_size,
            method="multi",
        )
        logger.info("  ✓ %s loaded: %s rows", table_name, f"{len(df):,}")


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  FinTech Fraud Detection — Synthetic Data Generator")
    print("=" * 70)

    # --- Generate base entities ---
    customers_df = generate_customers(50_000)
    merchants_df = generate_merchants(5_000)
    txn_df = generate_transactions(customers_df, merchants_df, 1_000_000)

    # --- Inject fraud patterns ---
    txn_df = inject_geographic_jumps(txn_df, 500)
    txn_df = inject_velocity_bursts(txn_df, 800)
    txn_df = inject_account_takeover(txn_df, customers_df, 400)
    txn_df = inject_card_testing(txn_df, 600)

    # --- Summary ---
    total = len(txn_df)
    fraud_count = txn_df["is_fraud"].sum()
    fraud_rate = fraud_count / total * 100
    print(f"\n{'─' * 50}")
    print(f"  Total transactions : {total:>12,}")
    print(f"  Fraud transactions : {fraud_count:>12,}")
    print(f"  Fraud rate         : {fraud_rate:>11.2f}%")
    print(f"{'─' * 50}")

    by_type = txn_df[txn_df["is_fraud"]].groupby("fraud_type").size()
    print("\n  Fraud breakdown:")
    for ftype, fcount in by_type.items():
        print(f"    {ftype:<25s} {fcount:>8,}")

    # --- Load to DB ---
    engine = _get_engine()
    load_to_postgres(customers_df, merchants_df, txn_df, engine)

    print("\n✅ Data generation and database load complete.")
