-- ============================================================
-- Geographic Jump Detection (Impossible Travel)
-- ============================================================
-- Detects when the same card is used at physical terminals in
-- two different countries within 30 minutes — a physically
-- impossible scenario indicating card compromise.
--
-- Business context: this is a high-confidence fraud signal.
-- Legitimate customers cannot physically travel between
-- countries in under 30 minutes, so any such pattern on a
-- POS or ATM terminal is almost certainly fraudulent.
-- ============================================================

WITH physical_txns AS (
    -- Filter to only physical terminal types (POS & ATM)
    SELECT
        transaction_id,
        customer_id,
        transaction_ts,
        amount_gbp,
        terminal_type,
        terminal_country,
        merchant_category,
        is_fraud
    FROM analytics.transactions_analytics_ready
    WHERE terminal_type IN ('pos', 'atm')
),

with_previous AS (
    SELECT
        transaction_id,
        customer_id,
        transaction_ts,
        amount_gbp,
        terminal_type,
        terminal_country,
        merchant_category,
        is_fraud,

        -- Previous transaction details for the same customer
        LAG(transaction_id) OVER (
            PARTITION BY customer_id ORDER BY transaction_ts
        ) AS prev_transaction_id,

        LAG(terminal_country) OVER (
            PARTITION BY customer_id ORDER BY transaction_ts
        ) AS prev_country,

        LAG(transaction_ts) OVER (
            PARTITION BY customer_id ORDER BY transaction_ts
        ) AS prev_ts

    FROM physical_txns
)

SELECT
    transaction_id          AS current_transaction_id,
    prev_transaction_id,
    customer_id,
    transaction_ts          AS current_ts,
    prev_ts,
    terminal_country        AS current_country,
    prev_country,
    amount_gbp,
    merchant_category,
    is_fraud,

    ROUND(
        EXTRACT(EPOCH FROM (transaction_ts - prev_ts)) / 60, 1
    ) AS minutes_between_txns,

    'GEOGRAPHIC_JUMP' AS alert_type,
    'HIGH'            AS alert_severity,

    FORMAT(
        'Card used in %s then %s within %.1f minutes',
        prev_country,
        terminal_country,
        EXTRACT(EPOCH FROM (transaction_ts - prev_ts)) / 60
    ) AS alert_description

FROM with_previous
WHERE
    prev_country IS NOT NULL
    AND terminal_country != prev_country
    AND EXTRACT(EPOCH FROM (transaction_ts - prev_ts)) / 60 < 30
ORDER BY minutes_between_txns ASC;
