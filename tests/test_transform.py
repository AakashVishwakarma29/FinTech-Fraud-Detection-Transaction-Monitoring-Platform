"""
Tests for ETL Transform Module
================================
Validates the core data-cleaning operations: deduplication,
currency conversion, null handling, and text standardisation.
"""

import pandas as pd
import pytest

# Insert project root so imports work when running pytest from project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.transform import transform_pipeline


@pytest.fixture
def sample_raw_data():
    """Create a minimal raw transaction DataFrame for testing."""
    return pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t1"],  # t1 is duplicated
        "customer_id": ["c1", "c1", "c2", "c1"],
        "merchant_id": ["m1", "m2", "m1", "m1"],
        "amount_gbp": [100.0, None, 50.0, 100.0],
        "original_currency": ["USD", "GBP", "EUR", "USD"],
        "original_amount": [100.0, 200.0, 50.0, 100.0],
        "transaction_ts": [
            "2025-01-01 10:00:00",
            "2025-01-01 11:00:00",
            "2025-01-01 12:00:00",
            "2025-01-01 10:00:00",
        ],
        "terminal_type": ["POS", " Online ", "ATM", "POS"],
        "terminal_country": ["GB", None, "DE", "GB"],
        "merchant_category": ["retail", "travel", "ELECTRONICS", "retail"],
        "is_declined": [False, False, False, False],
        "is_disputed": [False, False, False, False],
        "is_fraud": [False, False, False, False],
    })


def test_duplicate_removal(sample_raw_data):
    """Verify that duplicate transaction_ids are removed."""
    df, audit = transform_pipeline(sample_raw_data)
    assert df["transaction_id"].nunique() == len(df)
    assert audit["duplicates_removed"] == 1


def test_currency_conversion_usd_to_gbp(sample_raw_data):
    """Verify USD 100 converts to GBP 79.00 (rate: 0.79)."""
    df, _ = transform_pipeline(sample_raw_data)
    row = df[df["transaction_id"] == "t1"].iloc[0]
    # USD 100 * 0.79 = 79.00
    assert row["amount_gbp"] == pytest.approx(79.0, abs=0.01)


def test_currency_conversion_eur_to_gbp(sample_raw_data):
    """Verify EUR 50 converts to GBP 43.00 (rate: 0.86)."""
    df, _ = transform_pipeline(sample_raw_data)
    row = df[df["transaction_id"] == "t3"].iloc[0]
    # EUR 50 * 0.86 = 43.00
    assert row["amount_gbp"] == pytest.approx(43.0, abs=0.01)


def test_no_null_amounts_after_transform(sample_raw_data):
    """Verify no null values remain in amount_gbp."""
    df, _ = transform_pipeline(sample_raw_data)
    assert df["amount_gbp"].isna().sum() == 0


def test_terminal_type_lowercase(sample_raw_data):
    """Verify terminal_type is always lowercase and stripped."""
    df, _ = transform_pipeline(sample_raw_data)
    for val in df["terminal_type"]:
        assert val == val.lower().strip()


def test_merchant_category_titlecase(sample_raw_data):
    """Verify merchant_category is title-cased after transform."""
    df, _ = transform_pipeline(sample_raw_data)
    for val in df["merchant_category"]:
        assert val == val.title().strip()


def test_online_terminal_country_filled(sample_raw_data):
    """Verify null terminal_country is filled with 'GB' for online terminals."""
    df, _ = transform_pipeline(sample_raw_data)
    online_rows = df[df["terminal_type"] == "online"]
    assert (online_rows["terminal_country"] == "GB").all()
