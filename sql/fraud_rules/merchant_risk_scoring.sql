-- ============================================================
-- Merchant Risk Scoring
-- ============================================================
-- Computes a composite risk score for each merchant based on
-- dispute rates, decline rates, and confirmed fraud counts.
--
-- Business context: merchants with disproportionately high
-- dispute and fraud rates may be complicit in fraud, operating
-- as fronts, or have inadequate security controls.  This query
-- identifies the top 50 riskiest merchants for investigation.
-- ============================================================

WITH merchant_stats AS (
    SELECT
        merchant_id,
        merchant_name,
        merchant_category,

        COUNT(*)                                    AS total_txns,
        ROUND(SUM(amount_gbp), 2)                   AS total_volume_gbp,
        ROUND(AVG(amount_gbp), 2)                   AS avg_txn_value,

        COUNT(*) FILTER (WHERE is_disputed = TRUE)   AS disputed_count,
        COUNT(*) FILTER (WHERE is_declined = TRUE)   AS declined_count,
        COUNT(*) FILTER (WHERE is_fraud    = TRUE)   AS confirmed_fraud_count,

        -- Dispute rate as percentage
        ROUND(
            COUNT(*) FILTER (WHERE is_disputed = TRUE) * 100.0 / COUNT(*),
            2
        ) AS dispute_rate_pct,

        -- Decline rate as percentage
        ROUND(
            COUNT(*) FILTER (WHERE is_declined = TRUE) * 100.0 / COUNT(*),
            2
        ) AS decline_rate_pct

    FROM analytics.transactions_analytics_ready
    GROUP BY merchant_id, merchant_name, merchant_category
    HAVING COUNT(*) >= 50
),

risk_scored AS (
    SELECT
        *,
        -- Composite risk score: weighted blend of dispute, decline, and fraud
        ROUND(
            (dispute_rate_pct * 0.5)
            + (decline_rate_pct * 0.3)
            + (confirmed_fraud_count * 100.0 / total_txns * 0.2),
            2
        ) AS composite_risk_score,

        -- Risk quartile (1 = highest dispute rate)
        NTILE(4) OVER (ORDER BY dispute_rate_pct DESC) AS risk_quartile

    FROM merchant_stats
)

SELECT
    merchant_id,
    merchant_name,
    merchant_category,
    total_txns,
    total_volume_gbp,
    avg_txn_value,
    disputed_count,
    declined_count,
    confirmed_fraud_count,
    dispute_rate_pct,
    decline_rate_pct,
    composite_risk_score,
    risk_quartile,

    CASE risk_quartile
        WHEN 1 THEN 'CRITICAL'
        WHEN 2 THEN 'HIGH'
        WHEN 3 THEN 'MEDIUM'
        WHEN 4 THEN 'LOW'
    END AS risk_tier

FROM risk_scored
ORDER BY composite_risk_score DESC
LIMIT 50;
