"""
ML Model Evaluation
=====================
Loads the trained XGBoost fraud model and generates:
  - Confusion matrix
  - Feature importance plot (top 15)
  - Precision/Recall/F1 at threshold 0.5
  - Best threshold by F1 score (swept 0.1–0.9)
  - Saved charts to models/ directory
"""

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

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


def load_model_and_data():
    """
    Load the trained XGBoost model and saved test predictions.

    Business context: evaluation is decoupled from training so the
    model can be re-evaluated with different thresholds without
    re-training (which is expensive on 1M+ rows).
    """
    model_path = MODELS_DIR / "xgb_fraud_v1.pkl"
    test_path = MODELS_DIR / "test_predictions.parquet"

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading test predictions from %s", test_path)
    test_df = pd.read_parquet(test_path)

    return model, test_df


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Generate and save a confusion matrix heatmap.

    Business context: the confusion matrix reveals the trade-off
    between catching fraud (recall) and false positives that
    frustrate legitimate customers.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Fraud Detection — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Confusion matrix saved to %s", save_path)


def plot_feature_importance(model, save_path, top_n=15):
    """
    Plot the top N most important features by XGBoost gain.

    Business context: understanding which features drive fraud
    predictions helps analysts validate the model against known
    fraud typologies and identify new patterns.
    """
    importance = model.feature_importances_
    feature_names = FEATURE_COLUMNS

    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", n_colors=top_n)
    bars = ax.barh(range(top_n), top_importance[::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Features — XGBoost Fraud Classifier",
        fontsize=14, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Feature importance chart saved to %s", save_path)


def find_best_threshold(y_true, y_prob, low=0.1, high=0.9, steps=81):
    """
    Sweep thresholds to find the one maximising F1 score.

    Business context: the default 0.5 threshold is rarely optimal
    for imbalanced fraud data. Sweeping finds the best balance
    between precision (avoiding false alarms) and recall (catching
    all fraud).

    Returns:
        Tuple of (best_threshold, best_f1).
    """
    thresholds = np.linspace(low, high, steps)
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return round(best_thresh, 3), round(best_f1, 4)


def evaluate():
    """Run full model evaluation suite."""
    model, test_df = load_model_and_data()

    y_true = test_df["y_true"].values
    y_prob = test_df["y_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION — DETAILED RESULTS")
    print("=" * 60)

    # Metrics at default threshold
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  Threshold = 0.5:")
    print(f"    Precision : {p:.4f}")
    print(f"    Recall    : {r:.4f}")
    print(f"    F1 Score  : {f1:.4f}")

    # Optimal threshold
    best_thresh, best_f1 = find_best_threshold(y_true, y_prob)
    print(f"\n  Best threshold by F1: {best_thresh}")
    print(f"    F1 at best threshold: {best_f1}")

    y_pred_opt = (y_prob >= best_thresh).astype(int)
    p_opt = precision_score(y_true, y_pred_opt, zero_division=0)
    r_opt = recall_score(y_true, y_pred_opt, zero_division=0)
    print(f"    Precision : {p_opt:.4f}")
    print(f"    Recall    : {r_opt:.4f}")

    print("=" * 60)

    # Generate plots
    plot_confusion_matrix(
        y_true, y_pred,
        MODELS_DIR / "confusion_matrix.png",
    )
    plot_feature_importance(
        model,
        MODELS_DIR / "feature_importance.png",
        top_n=15,
    )

    print("\n✅ Evaluation complete. Charts saved to ml/models/")


if __name__ == "__main__":
    evaluate()
