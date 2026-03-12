"""
Full Pipeline Orchestrator
===========================
Runs the entire fraud detection pipeline end-to-end:
  1. Generate synthetic data → load to database
  2. Extract → Transform → Feature Engineering → Load analytics table
  3. Train ML model
  4. Evaluate ML model

Supports both PostgreSQL (Supabase) and SQLite (local) backends.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from db_config import get_engine, init_sqlite_schema


def run_pipeline(n_customers=5_000, n_merchants=500, n_transactions=50_000):
    """
    Run the complete pipeline with configurable data sizes.

    Default sizes are reduced for fast demo runs. For full production
    scale, use n_customers=50_000, n_merchants=5_000, n_transactions=1_000_000.
    """
    start = time.time()

    # ── Step 0: Database setup ──────────────────────────────────────
    print("=" * 60)
    print("  FinTech Fraud Detection — Full Pipeline")
    print("=" * 60)

    engine = get_engine()
    dialect = engine.dialect.name
    print(f"\n  Database backend: {dialect.upper()}")

    if dialect == "sqlite":
        init_sqlite_schema(engine)
        print("  SQLite schema initialized")

    # ── Step 1: Generate synthetic data ─────────────────────────────
    print(f"\n{'─' * 60}")
    print("  STEP 1: Generating synthetic data …")
    print(f"{'─' * 60}")

    from data.generate_data import (
        generate_customers,
        generate_merchants,
        generate_transactions,
        inject_geographic_jumps,
        inject_velocity_bursts,
        inject_account_takeover,
        inject_card_testing,
        load_to_postgres,
    )

    customers_df = generate_customers(n_customers)
    merchants_df = generate_merchants(n_merchants)
    txn_df = generate_transactions(customers_df, merchants_df, n_transactions)

    # Inject fraud patterns (scaled proportionally)
    scale = n_transactions / 1_000_000
    txn_df = inject_geographic_jumps(txn_df, max(10, int(500 * scale)))
    txn_df = inject_velocity_bursts(txn_df, max(10, int(800 * scale)))
    txn_df = inject_account_takeover(txn_df, customers_df, max(10, int(400 * scale)))
    txn_df = inject_card_testing(txn_df, max(10, int(600 * scale)))

    fraud_count = txn_df["is_fraud"].sum()
    total = len(txn_df)
    print(f"\n  Total transactions: {total:,}")
    print(f"  Fraud transactions: {fraud_count:,} ({fraud_count/total*100:.1f}%)")

    # Load to database
    load_to_postgres(customers_df, merchants_df, txn_df, engine, batch_size=5_000)

    # ── Step 2: ETL Pipeline ────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  STEP 2: Running ETL pipeline …")
    print(f"{'─' * 60}")

    from etl.transform import transform_pipeline
    from etl.feature_engineering import engineer_features
    from etl.load import load_to_analytics

    # For SQLite, we read directly from the loaded tables
    import pandas as pd
    from sqlalchemy import text

    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT
                t.*, 
                c.full_name AS customer_name, c.email AS customer_email,
                c.date_of_birth, c.account_opened, c.region, c.risk_score,
                m.merchant_name, m.category AS merchant_category,
                m.country_code AS merchant_country, m.is_high_risk
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            JOIN merchants m ON t.merchant_id = m.merchant_id
        """, conn)

    print(f"  Extracted: {len(df):,} rows")

    df, audit = transform_pipeline(df)
    print(f"  Transformed: {len(df):,} rows (removed {audit['duplicates_removed']} dupes)")

    df = engineer_features(df)
    print(f"  Features engineered: {len(df.columns)} columns")

    load_to_analytics(df, engine)

    # ── Step 3: Train ML model ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  STEP 3: Training ML model …")
    print(f"{'─' * 60}")

    from ml.train_model import train_fraud_model, load_training_data, MODELS_DIR

    # Load from analytics table directly
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    analytics_df = pd.read_sql("SELECT * FROM transactions_analytics_ready", engine)

    le_cat = LabelEncoder()
    le_reg = LabelEncoder()
    analytics_df["merchant_category_enc"] = le_cat.fit_transform(
        analytics_df["merchant_category"].fillna("Unknown").astype(str)
    )
    analytics_df["region_enc"] = le_reg.fit_transform(
        analytics_df["region"].fillna("Unknown").astype(str)
    )
    analytics_df["is_fraud"] = analytics_df["is_fraud"].astype(int)
    analytics_df["is_high_risk"] = analytics_df["is_high_risk"].astype(int)

    import joblib
    joblib.dump(le_cat, MODELS_DIR / "label_encoder_category.pkl")
    joblib.dump(le_reg, MODELS_DIR / "label_encoder_region.pkl")

    train_fraud_model(analytics_df)

    # ── Step 4: Evaluate model ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  STEP 4: Evaluating ML model …")
    print(f"{'─' * 60}")

    from ml.evaluate_model import evaluate
    evaluate()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Demo mode with reduced data for fast execution
    run_pipeline(
        n_customers=5_000,
        n_merchants=500,
        n_transactions=50_000,
    )
