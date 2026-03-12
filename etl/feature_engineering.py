"""
ETL Feature Engineering Module
================================
Engineers temporal, velocity, geographic, and account-age features
from cleaned transaction data for ML training and analytics.

Creates 11 new columns that capture fraud-indicative signals.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fraud-detection features from cleaned transaction data.

    Business context: raw transaction fields alone are poor predictors;
    engineered features like time-since-last-transaction, rolling spend,
    and geographic mismatches capture the behavioural signals that
    distinguish legitimate from fraudulent activity.

    Features created:
      - hour_of_day: hour component of transaction timestamp
      - day_of_week: 0=Monday … 6=Sunday
      - is_weekend: 1 if Saturday/Sunday
      - is_night: 1 if hour between 22:00–05:00
      - minutes_since_last_txn: per-customer time gap (9999 for first)
      - rolling_7d_spend: 7-day rolling sum of amount_gbp per customer
      - is_foreign_merchant: 1 if merchant not based in GB
      - is_country_mismatch: terminal country ≠ merchant country
      - account_age_days: days since account was opened
      - is_new_account: 1 if account < 90 days old
      - amount_vs_median_ratio: txn amount / customer median (clipped ≥ 1)

    Args:
        df: Cleaned DataFrame from transform stage.

    Returns:
        DataFrame with new feature columns appended.
    """
    logger.info("Feature engineering started — %s rows", f"{len(df):,}")

    # Sort for correct window calculations
    df = df.sort_values(["customer_id", "transaction_ts"]).reset_index(drop=True)

    # Ensure timestamp is proper datetime
    df["transaction_ts"] = pd.to_datetime(df["transaction_ts"], utc=True)

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------
    df["hour_of_day"] = df["transaction_ts"].dt.hour
    df["day_of_week"] = df["transaction_ts"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"].isin([5, 6])).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    logger.info("  [1/5] Temporal features: hour, day, weekend, night")

    # ------------------------------------------------------------------
    # Velocity features
    # ------------------------------------------------------------------
    df["_prev_ts"] = df.groupby("customer_id")["transaction_ts"].shift(1)
    df["minutes_since_last_txn"] = (
        (df["transaction_ts"] - df["_prev_ts"]).dt.total_seconds() / 60
    ).fillna(9999).round(2)
    df = df.drop(columns=["_prev_ts"])
    logger.info("  [2/5] Velocity feature: minutes_since_last_txn")

    # Rolling 7-day spend per customer
    df = df.set_index("transaction_ts")
    df["rolling_7d_spend"] = (
        df.groupby("customer_id")["amount_gbp"]
        .rolling("7D", min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index()
    logger.info("  [3/5] Rolling 7-day spend calculated")

    # ------------------------------------------------------------------
    # Geographic features
    # ------------------------------------------------------------------
    df["is_foreign_merchant"] = (df["merchant_country"] != "GB").astype(int)
    df["is_country_mismatch"] = (
        df["terminal_country"] != df["merchant_country"]
    ).astype(int)
    logger.info("  [4/5] Geographic features: foreign merchant, country mismatch")

    # ------------------------------------------------------------------
    # Account-age features
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    df["account_opened"] = pd.to_datetime(df["account_opened"], utc=True)
    df["account_age_days"] = (now - df["account_opened"]).dt.days
    df["is_new_account"] = (df["account_age_days"] < 90).astype(int)
    logger.info("  [5/5] Account features: age, is_new_account")

    # ------------------------------------------------------------------
    # Ratio features
    # ------------------------------------------------------------------
    customer_medians = df.groupby("customer_id")["amount_gbp"].transform("median")
    df["amount_vs_median_ratio"] = (df["amount_gbp"] / customer_medians).clip(lower=1).round(4)
    # Handle edge case where median could be 0
    df["amount_vs_median_ratio"] = df["amount_vs_median_ratio"].replace(
        [np.inf, -np.inf], 1.0
    ).fillna(1.0)
    logger.info("  Ratio feature: amount_vs_median_ratio")

    logger.info("✓ Feature engineering complete: %s columns", len(df.columns))
    return df


if __name__ == "__main__":
    print("Feature engineering module — run as part of the ETL pipeline.")
    print("Usage: from etl.feature_engineering import engineer_features")
