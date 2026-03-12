-- ============================================================
-- Velocity Anomaly Detection
-- ============================================================
-- Identifies transactions whose spend significantly deviates from
-- a customer's rolling 7-day average, using z-scores and ratios.
--
-- Business context: sudden spending spikes (e.g. 10× normal) are
-- a strong indicator of compromised credentials or stolen cards.
-- ============================================================

WITH customer_rolling AS (
    SELECT
        transaction_id,
        customer_id,
        transaction_ts,
        amount_gbp,
        merchant_category,
        terminal_country,
        is_fraud,

        -- Rolling 7-day avg EXCLUDING current transaction
        -- 168 hours ≈ 7 days of hourly-granularity rows
        AVG(amount_gbp) OVER (
            PARTITION BY customer_id
            ORDER BY transaction_ts
            ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS rolling_7d_avg,

        -- Rolling 7-day stddev EXCLUDING current transaction
        STDDEV(amount_gbp) OVER (
            PARTITION BY customer_id
            ORDER BY transaction_ts
            ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS rolling_7d_stddev

    FROM analytics.transactions_analytics_ready
),

scored AS (
    SELECT
        transaction_id,
        customer_id,
        transaction_ts,
        amount_gbp,
        merchant_category,
        terminal_country,
        is_fraud,
        rolling_7d_avg,
        rolling_7d_stddev,

        -- Spend ratio: how many multiples of average?
        ROUND(amount_gbp / NULLIF(rolling_7d_avg, 0), 2)
            AS spend_ratio,

        -- Z-score: how many standard deviations from mean?
        ROUND(
            (amount_gbp - rolling_7d_avg) / NULLIF(rolling_7d_stddev, 0), 2
        ) AS z_score

    FROM customer_rolling
    WHERE rolling_7d_avg IS NOT NULL
)

SELECT
    transaction_id,
    customer_id,
    transaction_ts,
    amount_gbp,
    merchant_category,
    terminal_country,
    is_fraud,
    ROUND(rolling_7d_avg, 2)    AS rolling_7d_avg,
    ROUND(rolling_7d_stddev, 2) AS rolling_7d_stddev,
    spend_ratio,
    z_score,

    'VELOCITY_ANOMALY' AS alert_type,

    CASE
        WHEN spend_ratio > 10 OR z_score > 5  THEN 'HIGH'
        WHEN spend_ratio > 5  OR z_score > 3  THEN 'MEDIUM'
        ELSE 'LOW'
    END AS alert_severity

FROM scored
WHERE spend_ratio > 3.0
ORDER BY spend_ratio DESC;
