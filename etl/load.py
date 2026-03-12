"""
ETL Load Module
================
Writes the fully-transformed, feature-enriched DataFrame to the
analytics schema in PostgreSQL for downstream consumption by
SQL fraud rules, Tableau dashboards, and ML training.
"""

import logging
import os

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

    Business context: centralised engine factory ensures consistent
    database access across all pipeline stages, with automatic
    fallback to SQLite for local development.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from db_config import get_engine as _get_engine
    return _get_engine()


def load_to_analytics(df: pd.DataFrame, engine) -> None:
    """
    Write the analytics-ready DataFrame to the analytics table.

    Business context: the transactions_analytics_ready table serves as
    the single source of truth for all downstream consumers — SQL fraud
    rule queries, Tableau dashboards, and the ML training pipeline.
    Replacing the full table on each run ensures consistency.

    Supports both PostgreSQL (analytics schema) and SQLite (flat).

    Args:
        df: Feature-enriched DataFrame from the engineering stage.
        engine: SQLAlchemy engine instance.
    """
    target_table = "transactions_analytics_ready"
    is_sqlite = engine.dialect.name == "sqlite"
    target_schema = None if is_sqlite else "analytics"
    display_name = target_table if is_sqlite else f"analytics.{target_table}"

    logger.info("Loading %s rows → %s …", f"{len(df):,}", display_name)

    # SQLite doesn't support method='multi'
    to_sql_kwargs = {
        "name": target_table,
        "con": engine,
        "if_exists": "replace",
        "index": False,
        "chunksize": 10_000,
    }
    if target_schema:
        to_sql_kwargs["schema"] = target_schema
    if not is_sqlite:
        to_sql_kwargs["method"] = "multi"

    df.to_sql(**to_sql_kwargs)

    logger.info("✓ Load complete: %s rows written to %s", f"{len(df):,}", display_name)


if __name__ == "__main__":
    print("Load module — run as part of the ETL pipeline.")
    print("Usage: from etl.load import load_to_analytics, get_engine")
