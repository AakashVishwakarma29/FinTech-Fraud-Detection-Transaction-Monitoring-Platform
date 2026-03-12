"""
Tests for ETL Feature Engineering Module
==========================================
Validates that engineered features have correct value ranges
and business-logic constraints.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

# Insert project root so imports work when running pytest from project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.feature_engineering import engineer_features


@pytest.fixture
def sample_clean_data():
    """Create a minimal cleaned DataFrame for feature engineering tests."""
    now = datetime.now(timezone.utc)
    return pd.DataFrame({
        "transaction_id": [f"t{i}" for i in range(10)],
        "customer_id": ["c1"] * 5 + ["c2"] * 5,
        "merchant_id": ["m1"] * 10,
        "amount_gbp": [50.0, 120.0, 30.0, 200.0, 15.0,
                        80.0, 45.0, 300.0, 10.0, 60.0],
        "original_currency": ["GBP"] * 10,
        "original_amount": [50.0, 120.0, 30.0, 200.0, 15.0,
                             80.0, 45.0, 300.0, 10.0, 60.0],
        "transaction_ts": pd.to_datetime([
            now - timedelta(days=10),
            now - timedelta(days=9),
            now - timedelta(days=8),
            now - timedelta(days=7),
            now - timedelta(days=6),
            now - timedelta(days=5),
            now - timedelta(days=4),
            now - timedelta(days=3),
            now - timedelta(days=2),
            now - timedelta(days=1),
        ], utc=True),
        "terminal_type": ["pos"] * 10,
        "terminal_country": ["GB"] * 8 + ["FR", "GB"],
        "merchant_category": ["Retail"] * 10,
        "merchant_country": ["GB"] * 10,
        "account_opened": [
            (now - timedelta(days=500)).date(),
            (now - timedelta(days=500)).date(),
            (now - timedelta(days=500)).date(),
            (now - timedelta(days=500)).date(),
            (now - timedelta(days=500)).date(),
            (now - timedelta(days=30)).date(),   # New account
            (now - timedelta(days=30)).date(),
            (now - timedelta(days=30)).date(),
            (now - timedelta(days=30)).date(),
            (now - timedelta(days=30)).date(),
        ],
        "is_fraud": [False] * 10,
        "risk_score": [2.5] * 10,
    })


def test_hour_of_day_range(sample_clean_data):
    """hour_of_day must be between 0 and 23."""
    df = engineer_features(sample_clean_data)
    assert df["hour_of_day"].between(0, 23).all()


def test_day_of_week_range(sample_clean_data):
    """day_of_week must be between 0 (Monday) and 6 (Sunday)."""
    df = engineer_features(sample_clean_data)
    assert df["day_of_week"].between(0, 6).all()


def test_is_weekend_binary(sample_clean_data):
    """is_weekend must only contain 0 or 1."""
    df = engineer_features(sample_clean_data)
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_is_night_binary(sample_clean_data):
    """is_night must only contain 0 or 1."""
    df = engineer_features(sample_clean_data)
    assert set(df["is_night"].unique()).issubset({0, 1})


def test_minutes_since_last_txn_not_negative(sample_clean_data):
    """minutes_since_last_txn must never be negative."""
    df = engineer_features(sample_clean_data)
    assert (df["minutes_since_last_txn"] >= 0).all()


def test_amount_vs_median_ratio_not_below_zero(sample_clean_data):
    """amount_vs_median_ratio must never be below 0."""
    df = engineer_features(sample_clean_data)
    assert (df["amount_vs_median_ratio"] >= 0).all()


def test_amount_vs_median_ratio_clipped_at_one(sample_clean_data):
    """amount_vs_median_ratio is clipped to minimum 1.0."""
    df = engineer_features(sample_clean_data)
    assert (df["amount_vs_median_ratio"] >= 1.0).all()


def test_is_foreign_merchant_binary(sample_clean_data):
    """is_foreign_merchant must only contain 0 or 1."""
    df = engineer_features(sample_clean_data)
    assert set(df["is_foreign_merchant"].unique()).issubset({0, 1})


def test_is_new_account_flag(sample_clean_data):
    """Verify is_new_account correctly identifies accounts < 90 days old."""
    df = engineer_features(sample_clean_data)
    # c2 has account_opened 30 days ago → is_new_account should be 1
    c2_rows = df[df["customer_id"] == "c2"]
    assert (c2_rows["is_new_account"] == 1).all()
    # c1 has account_opened 500 days ago → is_new_account should be 0
    c1_rows = df[df["customer_id"] == "c1"]
    assert (c1_rows["is_new_account"] == 0).all()


def test_account_age_days_positive(sample_clean_data):
    """account_age_days should always be positive."""
    df = engineer_features(sample_clean_data)
    assert (df["account_age_days"] > 0).all()
