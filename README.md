# FinTech Fraud Detection & Transaction Monitoring Platform

A production-grade fraud detection system built for UK financial transactions. Combines advanced SQL analytics, a full ETL pipeline, and an XGBoost ML classifier to detect and monitor fraudulent activity across 1 million synthetic transactions.

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Synthetic Data │────▶│  PostgreSQL   │────▶│    ETL Pipeline      │
│  (Faker + NumPy)│     │  (Raw Tables) │     │ Extract → Transform  │
└─────────────────┘     └──────────────┘     │ → Feature Eng → Load │
                                              └──────────┬───────────┘
                                                         │
                              ┌───────────────────────────┼──────────────┐
                              ▼                           ▼              ▼
                    ┌──────────────────┐     ┌────────────────┐  ┌──────────┐
                    │  SQL Fraud Rules │     │  ML Classifier │  │ Tableau  │
                    │  (Window Funcs)  │     │  (XGBoost)     │  │  Views   │
                    └──────────────────┘     └────────────────┘  └──────────┘
```

## Tech Stack

| Layer            | Technology                                    |
|------------------|-----------------------------------------------|
| Language         | Python 3.10+                                  |
| Database         | PostgreSQL 14+                                |
| Data Generation  | Faker, NumPy, Pandas                          |
| ETL              | Pandas, SQLAlchemy                            |
| SQL Analytics    | Window Functions, CTEs, Aggregate Scoring     |
| Machine Learning | XGBoost, scikit-learn                         |
| Visualisation    | Tableau-ready SQL views, Matplotlib, Seaborn  |
| Testing          | pytest                                        |

## Project Structure

```
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── generate_data.py          # Synthetic data + fraud injection + DB load
├── etl/
│   ├── extract.py                # Pull data from PostgreSQL
│   ├── transform.py              # Dedup, fill nulls, normalise currencies
│   ├── feature_engineering.py    # Temporal, velocity & risk features
│   └── load.py                   # Write analytics-ready table
├── sql/
│   ├── schema/
│   │   └── 01_create_tables.sql  # Core DDL
│   ├── fraud_rules/
│   │   ├── velocity_checks.sql   # Spend anomaly detection
│   │   ├── geographic_jumps.sql  # Impossible-travel detection
│   │   └── merchant_risk_scoring.sql  # Composite merchant risk
│   └── analytics/
│       └── create_views.sql      # Executive summary & risk matrix views
├── ml/
│   ├── train_model.py            # XGBoost training pipeline
│   ├── evaluate_model.py         # Model evaluation & feature importance
│   └── models/                   # Serialised model artefacts
├── dashboards/                   # Tableau workbooks (future)
├── notebooks/                    # Exploratory analysis
└── tests/
    ├── test_transform.py
    └── test_feature_engineering.py
```

## Fraud Patterns Injected

| Pattern            | Count | Description                                          |
|--------------------|-------|------------------------------------------------------|
| Geographic Jump    | 500   | Same card used in two countries within 30 minutes    |
| Velocity Burst     | 800   | 5–9 txns within 60 minutes across multiple countries |
| Account Takeover   | 400   | High-value foreign txn on a brand-new account        |
| Card Testing       | 600   | Micro-transactions followed by one large purchase    |

Target overall fraud rate: **2–4%** of 1M transactions.

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/FinTech-Fraud-Detection-Transaction-Monitoring-Platform.git
cd FinTech-Fraud-Detection-Transaction-Monitoring-Platform
pip install -r requirements.txt

# 2. Configure database credentials
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# 3. Create database schema
psql -f sql/schema/01_create_tables.sql

# 4. Generate synthetic data & load to DB
python data/generate_data.py

# 5. Run ETL pipeline
python -c "
from etl.extract import extract_transactions
from etl.transform import transform_pipeline
from etl.feature_engineering import engineer_features
from etl.load import load_to_analytics, get_engine
engine = get_engine()
df = extract_transactions(engine)
df, audit = transform_pipeline(df)
df = engineer_features(df)
load_to_analytics(df, engine)
"

# 6. Create analytics views
psql -f sql/analytics/create_views.sql

# 7. Train ML model
python ml/train_model.py

# 8. Evaluate model
python ml/evaluate_model.py

# 9. Run tests
pytest tests/ -v
```

## SQL Fraud Rules

- **Velocity Checks** — Rolling 7-day spend z-scores with severity tiers
- **Geographic Jump Detection** — LAG-based impossible travel flagging
- **Merchant Risk Scoring** — Composite score from dispute, decline, and fraud rates

## ML Model

- **Algorithm**: XGBoost with `scale_pos_weight` for class imbalance
- **Features**: 16 engineered features (temporal, velocity, geographic, account-age)
- **Evaluation**: Precision-Recall AUC, confusion matrix, optimal F1 threshold sweep

## License

This project is for educational and portfolio purposes.
