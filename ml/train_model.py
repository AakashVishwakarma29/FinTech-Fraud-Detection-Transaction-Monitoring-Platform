"""
ML Model Training Pipeline
============================
Trains an XGBoost classifier for fraud detection on the analytics-ready
transaction data. Uses class-imbalance handling via scale_pos_weight,
early stopping, and stratified train/test splitting.

Outputs:
  - Trained model saved to models/xgb_fraud_v1.pkl
  - Classification report, ROC-AUC, and PR-AUC printed to console
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from xgboost import XGBClassifier

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root for model saving
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "amount_gbp",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "minutes_since_last_txn",
    "rolling_7d_spend",
    "is_foreign_merchant",
    "is_country_mismatch",
    "is_new_account",
    "amount_vs_median_ratio",
    "risk_score",
    "account_age_days",
    "is_high_risk",
    "merchant_category_enc",
    "region_enc",
]

TARGET = "is_fraud"


def _get_engine():
    """Build SQLAlchemy engine from .env credentials."""
    from urllib.parse import quote_plus
    password = quote_plus(os.getenv('DB_PASSWORD', ''))
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{password}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
        f"/{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(db_url)


def load_training_data(engine) -> pd.DataFrame:
    """
    Load analytics-ready data and encode categorical features.

    Business context: the ML model requires all-numeric inputs;
    merchant_category and region are label-encoded to preserve
    their categorical nature without one-hot explosion.
    """
    logger.info("Loading training data from analytics.transactions_analytics_ready …")
    df = pd.read_sql("SELECT * FROM analytics.transactions_analytics_ready", engine)
    logger.info("  Loaded %s rows", f"{len(df):,}")

    # Label-encode categoricals
    le_cat = LabelEncoder()
    le_reg = LabelEncoder()
    df["merchant_category_enc"] = le_cat.fit_transform(
        df["merchant_category"].fillna("Unknown").astype(str)
    )
    df["region_enc"] = le_reg.fit_transform(
        df["region"].fillna("Unknown").astype(str)
    )

    # Convert boolean target to int
    df[TARGET] = df[TARGET].astype(int)

    # Convert is_high_risk to int
    df["is_high_risk"] = df["is_high_risk"].astype(int)

    # Save encoders for inference
    joblib.dump(le_cat, MODELS_DIR / "label_encoder_category.pkl")
    joblib.dump(le_reg, MODELS_DIR / "label_encoder_region.pkl")
    logger.info("  Label encoders saved")

    return df


def train_fraud_model(df: pd.DataFrame) -> None:
    """
    Train XGBoost fraud classifier with class-imbalance handling.

    Business context: fraud is rare (~2-4%), so the model uses
    scale_pos_weight to up-weight fraud samples, aucpr as the
    evaluation metric (better than AUC-ROC for imbalanced data),
    and early stopping to prevent overfitting.
    """
    # Prepare features and target
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()

    # Handle any remaining NaN/inf in features
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Target distribution:\n%s", y.value_counts().to_string())

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("Train: %s | Test: %s", f"{len(X_train):,}", f"{len(X_test):,}")

    # Class imbalance ratio
    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info("Fraud ratio (neg/pos): %.2f", fraud_ratio)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        scale_pos_weight=fraud_ratio,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
    )

    logger.info("Training XGBoost (n_estimators=500, early_stopping=30) …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"  ROC-AUC Score      : {roc_auc:.4f}")
    print(f"  PR-AUC Score       : {pr_auc:.4f}")
    print(f"  Best iteration     : {model.best_iteration}")
    print("=" * 60)

    # Save model
    model_path = MODELS_DIR / "xgb_fraud_v1.pkl"
    joblib.dump(model, model_path)
    logger.info("✓ Model saved to %s", model_path)

    # Save test data for evaluation script
    test_data = X_test.copy()
    test_data["y_true"] = y_test.values
    test_data["y_prob"] = y_prob
    test_data.to_parquet(MODELS_DIR / "test_predictions.parquet", index=False)
    logger.info("✓ Test predictions saved for evaluation")


if __name__ == "__main__":
    print("=" * 60)
    print("  FinTech Fraud Detection — ML Training Pipeline")
    print("=" * 60)

    engine = _get_engine()
    data = load_training_data(engine)
    train_fraud_model(data)

    print("\n✅ Training pipeline complete.")
