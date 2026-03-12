"""
ETL Transform Module
=====================
Cleans, deduplicates, and normalises raw transaction data.

Pipeline steps:
  1. Deduplicate on transaction_id
  2. Fill missing values (amount, terminal_country)
  3. Normalise all amounts to GBP using fixed FX rates
  4. Standardise text fields (lowercase terminal_type, title-case category)
  5. Cast data types (timestamps, numerics)
"""

import logging
from typing import Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fixed FX rates to GBP
FX_TO_GBP = {
    "GBP": 1.0,
    "USD": 0.79,
    "EUR": 0.86,
    "JPY": 0.0052,
    "AUD": 0.51,
    "CAD": 0.58,
    "CHF": 0.89,
}


def transform_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Run the full cleansing and normalisation pipeline.

    Business context: raw transactional data contains duplicates from
    payment retries, missing fields from POS device failures, and
    multi-currency amounts that must be normalised to GBP for consistent
    analytics and ML training.

    Args:
        df: Raw extracted DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, audit dictionary with change counts).
    """
    audit = {}
    initial_rows = len(df)
    logger.info("Transform pipeline started — %s rows", f"{initial_rows:,}")

    # ------------------------------------------------------------------
    # STEP 1: Deduplicate on transaction_id
    # ------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    dups_removed = before - len(df)
    audit["duplicates_removed"] = dups_removed
    logger.info("  [1/5] Dedup: removed %s duplicates", f"{dups_removed:,}")

    # ------------------------------------------------------------------
    # STEP 2: Fill missing values
    # ------------------------------------------------------------------
    # amount_gbp: fill with merchant-category median
    null_amounts_before = df["amount_gbp"].isna().sum()
    if null_amounts_before > 0:
        category_medians = df.groupby("merchant_category")["amount_gbp"].transform("median")
        df["amount_gbp"] = df["amount_gbp"].fillna(category_medians)
        # Global fallback for any remaining nulls
        df["amount_gbp"] = df["amount_gbp"].fillna(df["amount_gbp"].median())
    audit["null_amounts_filled"] = null_amounts_before
    logger.info("  [2/5] Missing values: filled %s null amounts", null_amounts_before)

    # terminal_country: fill 'GB' for online terminal types
    null_countries_before = df["terminal_country"].isna().sum()
    online_mask = df["terminal_type"].str.lower().str.strip() == "online"
    df.loc[online_mask & df["terminal_country"].isna(), "terminal_country"] = "GB"
    # Remaining nulls: default to GB
    df["terminal_country"] = df["terminal_country"].fillna("GB")
    audit["null_countries_filled"] = null_countries_before
    logger.info("  [2/5] Missing values: filled %s null countries", null_countries_before)

    # ------------------------------------------------------------------
    # STEP 3: Currency normalisation
    # ------------------------------------------------------------------
    df["fx_rate"] = df["original_currency"].map(FX_TO_GBP).fillna(1.0)
    df["amount_gbp"] = (df["original_amount"] * df["fx_rate"]).round(2)
    df = df.drop(columns=["fx_rate"])
    audit["currency_normalised"] = True
    logger.info("  [3/5] Currency normalisation: recalculated all amounts to GBP")

    # ------------------------------------------------------------------
    # STEP 4: Standardise text fields
    # ------------------------------------------------------------------
    df["terminal_type"] = df["terminal_type"].str.lower().str.strip()
    df["merchant_category"] = df["merchant_category"].str.title().str.strip()
    audit["text_standardised"] = True
    logger.info("  [4/5] Text standardisation: terminal_type→lower, category→title")

    # ------------------------------------------------------------------
    # STEP 5: Cast data types
    # ------------------------------------------------------------------
    df["transaction_ts"] = pd.to_datetime(df["transaction_ts"], utc=True)
    df["amount_gbp"] = df["amount_gbp"].astype(float)
    audit["types_cast"] = True
    logger.info("  [5/5] Type casting: timestamps→UTC, amount_gbp→float")

    audit["final_rows"] = len(df)
    audit["rows_removed"] = initial_rows - len(df)
    logger.info("✓ Transform complete: %s → %s rows", f"{initial_rows:,}", f"{len(df):,}")
    return df, audit


if __name__ == "__main__":
    print("Transform module — run as part of the ETL pipeline.")
    print("Usage: from etl.transform import transform_pipeline")
