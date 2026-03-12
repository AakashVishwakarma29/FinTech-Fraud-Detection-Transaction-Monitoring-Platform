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
    Create a SQLAlchemy engine from environment variables.

    Business context: centralised engine factory ensures consistent
    database access across all pipeline stages.
    """
    from urllib.parse import quote_plus
    password = quote_plus(os.getenv('DB_PASSWORD', ''))
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{password}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
        f"/{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(db_url)


def load_to_analytics(df: pd.DataFrame, engine) -> None:
    """
    Write the analytics-ready DataFrame to the analytics schema.

    Business context: the analytics.transactions_analytics_ready table
    serves as the single source of truth for all downstream consumers —
    SQL fraud rule queries, Tableau dashboards, and the ML training
    pipeline. Replacing the full table on each run ensures consistency.

    Args:
        df: Feature-enriched DataFrame from the engineering stage.
        engine: SQLAlchemy engine instance.
    """
    target_table = "transactions_analytics_ready"
    target_schema = "analytics"

    logger.info(
        "Loading %s rows → %s.%s …",
        f"{len(df):,}", target_schema, target_table,
    )

    df.to_sql(
        target_table,
        engine,
        schema=target_schema,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        method="multi",
    )

    logger.info(
        "✓ Load complete: %s rows written to %s.%s",
        f"{len(df):,}", target_schema, target_table,
    )


if __name__ == "__main__":
    print("Load module — run as part of the ETL pipeline.")
    print("Usage: from etl.load import load_to_analytics, get_engine")
