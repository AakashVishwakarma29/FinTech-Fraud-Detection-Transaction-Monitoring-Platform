-- ============================================================
-- Analytics Views for Tableau & Executive Reporting
-- ============================================================
-- Two materialised views designed for direct Tableau connection:
--   1. executive_summary  — daily fraud KPIs
--   2. risk_matrix         — category × hour fraud heatmap
-- ============================================================

-- -----------------------------------------------------------
-- VIEW 1: Executive Summary (daily fraud KPIs)
-- -----------------------------------------------------------
-- Business context: provides the C-suite and compliance team
-- with a daily snapshot of transaction volume, fraud exposure,
-- and fraud rate trends for regulatory reporting and board decks.
-- -----------------------------------------------------------

CREATE OR REPLACE VIEW analytics.executive_summary AS
SELECT
    DATE(transaction_ts)                            AS txn_date,
    COUNT(*)                                        AS total_transactions,
    ROUND(SUM(amount_gbp), 2)                       AS total_volume_gbp,
    COUNT(*) FILTER (WHERE is_fraud = TRUE)          AS fraud_count,
    ROUND(
        SUM(amount_gbp) FILTER (WHERE is_fraud = TRUE), 2
    )                                               AS fraud_exposure_gbp,
    ROUND(
        COUNT(*) FILTER (WHERE is_fraud = TRUE) * 100.0 / COUNT(*), 2
    )                                               AS fraud_rate_pct
FROM analytics.transactions_analytics_ready
GROUP BY DATE(transaction_ts)
ORDER BY txn_date;


-- -----------------------------------------------------------
-- VIEW 2: Risk Matrix (category × hour heatmap)
-- -----------------------------------------------------------
-- Business context: fraud analysts use this heatmap to identify
-- which merchant categories are most vulnerable at specific
-- hours of the day, enabling targeted monitoring schedules.
-- -----------------------------------------------------------

CREATE OR REPLACE VIEW analytics.risk_matrix AS
SELECT
    merchant_category,
    EXTRACT(HOUR FROM transaction_ts)::INT           AS hour_of_day,
    COUNT(*)                                         AS transaction_count,
    COUNT(*) FILTER (WHERE is_fraud = TRUE)           AS fraud_count,
    ROUND(
        COUNT(*) FILTER (WHERE is_fraud = TRUE) * 100.0 / COUNT(*), 2
    )                                                AS fraud_rate_pct
FROM analytics.transactions_analytics_ready
GROUP BY merchant_category, EXTRACT(HOUR FROM transaction_ts)
HAVING COUNT(*) > 20
ORDER BY merchant_category, hour_of_day;
