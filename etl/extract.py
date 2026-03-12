"""
ETL Extract Module
==================
Pulls raw transactional data from PostgreSQL by joining the three core
tables (transactions, customers, merchants) into a single denormalised
DataFrame ready for transformation.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_engine():
    """
    Create a SQLAlchemy engine using the centralized db_config.

    Business context: centralised engine factory ensures all pipeline
    stages use consistent database credentials loaded from .env,
    with automatic fallback to SQLite for local development.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from db_config import get_engine as _get_engine
    return _get_engine()


EXTRACT_QUERY = """
SELECT
    t.transaction_id,
    t.customer_id,
    t.merchant_id,
    t.amount_gbp,
    t.original_currency,
    t.original_amount,
    t.transaction_ts,
    t.terminal_type,
    t.terminal_country,
    t.is_declined,
    t.is_disputed,
    t.is_fraud,
    t.fraud_type,
    c.full_name          AS customer_name,
    c.email              AS customer_email,
    c.date_of_birth,
    c.account_opened,
    c.region,
    c.risk_score,
    m.merchant_name,
    m.category           AS merchant_category,
    m.country_code       AS merchant_country,
    m.is_high_risk
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
JOIN merchants m ON t.merchant_id = m.merchant_id
"""


def extract_transactions(engine, batch_size: int = 100_000) -> pd.DataFrame:
    """
    Extract all transactions with customer and merchant details.

    Business context: the extract stage pulls a fully denormalised view
    of every transaction so downstream transforms can reference customer
    risk scores, merchant categories, and geographic information without
    additional database round-trips.

    Args:
        engine: SQLAlchemy engine instance.
        batch_size: number of rows per chunk (for memory efficiency).

    Returns:
        Single concatenated DataFrame of all transactions.
    """
    logger.info("Extracting transactions (batch_size=%s) …", f"{batch_size:,}")
    chunks = []
    batch_num = 0
    for chunk in pd.read_sql(EXTRACT_QUERY, engine, chunksize=batch_size):
        batch_num += 1
        chunks.append(chunk)
        logger.info("  Batch %d: %s rows read", batch_num, f"{len(chunk):,}")

    df = pd.concat(chunks, ignore_index=True)
    logger.info("✓ Extraction complete: %s total rows", f"{len(df):,}")
    return df


if __name__ == "__main__":
    print("Running extraction standalone …")
    eng = get_engine()
    result = extract_transactions(eng)
    print(f"Extracted {len(result):,} rows")
    print(result.dtypes)
