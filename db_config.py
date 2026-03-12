"""
Database Configuration & Connection Factory
=============================================
Provides a unified engine factory that supports both PostgreSQL (Supabase)
and SQLite (local development / demo) backends.

The connection falls back to SQLite automatically if PostgreSQL is
unreachable, ensuring the pipeline always works for local development.
"""

import logging
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SQLITE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"


def get_engine(force_sqlite=False):
    """
    Create a SQLAlchemy engine with automatic fallback.

    Business context: production systems use Supabase PostgreSQL, but
    for local development and demos, SQLite provides a zero-config
    alternative that lets the full pipeline run without external
    dependencies.

    Priority:
      1. PostgreSQL (from .env) if reachable
      2. SQLite fallback (local file)
    """
    if force_sqlite:
        logger.info("Using SQLite (forced): %s", SQLITE_PATH)
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(f"sqlite:///{SQLITE_PATH}", echo=False)

    # Try PostgreSQL first
    db_host = os.getenv("DB_HOST")
    if db_host:
        password = quote_plus(os.getenv("DB_PASSWORD", ""))
        db_url = (
            f"postgresql://{os.getenv('DB_USER')}:{password}"
            f"@{db_host}:{os.getenv('DB_PORT', '5432')}"
            f"/{os.getenv('DB_NAME', 'postgres')}"
        )
        try:
            engine = create_engine(db_url, connect_args={"connect_timeout": 5})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Connected to PostgreSQL: %s", db_host)
            return engine
        except Exception as e:
            logger.warning("PostgreSQL unreachable (%s), falling back to SQLite", str(e)[:80])

    # Fallback to SQLite
    logger.info("Using SQLite fallback: %s", SQLITE_PATH)
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{SQLITE_PATH}", echo=False)


def init_sqlite_schema(engine):
    """
    Create tables in SQLite (adapted from the PostgreSQL schema).

    Business context: SQLite doesn't support PostgreSQL-specific features
    like UUID defaults and schemas, so this provides an equivalent local
    schema for demo purposes.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                full_name TEXT,
                email TEXT UNIQUE,
                date_of_birth TEXT,
                account_opened TEXT,
                region TEXT,
                risk_score REAL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS merchants (
                merchant_id TEXT PRIMARY KEY,
                merchant_name TEXT,
                category TEXT,
                country_code TEXT,
                is_high_risk INTEGER DEFAULT 0
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT REFERENCES customers(customer_id),
                merchant_id TEXT REFERENCES merchants(merchant_id),
                amount_gbp REAL,
                original_currency TEXT,
                original_amount REAL,
                transaction_ts TEXT,
                terminal_type TEXT,
                terminal_country TEXT,
                is_declined INTEGER DEFAULT 0,
                is_disputed INTEGER DEFAULT 0,
                is_fraud INTEGER DEFAULT 0,
                fraud_type TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transactions_analytics_ready (
                transaction_id TEXT,
                customer_id TEXT,
                merchant_id TEXT,
                amount_gbp REAL,
                original_currency TEXT,
                original_amount REAL,
                transaction_ts TEXT,
                terminal_type TEXT,
                terminal_country TEXT,
                is_declined INTEGER,
                is_disputed INTEGER,
                is_fraud INTEGER,
                fraud_type TEXT,
                customer_name TEXT,
                customer_email TEXT,
                date_of_birth TEXT,
                account_opened TEXT,
                region TEXT,
                risk_score REAL,
                merchant_name TEXT,
                merchant_category TEXT,
                merchant_country TEXT,
                is_high_risk INTEGER,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                is_weekend INTEGER,
                is_night INTEGER,
                minutes_since_last_txn REAL,
                rolling_7d_spend REAL,
                is_foreign_merchant INTEGER,
                is_country_mismatch INTEGER,
                account_age_days INTEGER,
                is_new_account INTEGER,
                amount_vs_median_ratio REAL
            )
        """))
        conn.commit()
    logger.info("✓ SQLite schema initialized")


if __name__ == "__main__":
    engine = get_engine()
    dialect = engine.dialect.name
    print(f"Engine dialect: {dialect}")
    if dialect == "sqlite":
        init_sqlite_schema(engine)
        print("SQLite schema created.")
