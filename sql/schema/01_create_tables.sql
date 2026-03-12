-- ============================================================
-- FinTech Fraud Detection Platform — Database Schema
-- ============================================================
-- Creates the core transactional tables and analytics schema
-- for a UK-focused fraud detection & monitoring system.
-- ============================================================

-- Enable cryptographic UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -----------------------------------------------------------
-- 1. CUSTOMERS
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    customer_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name       VARCHAR(100),
    email           VARCHAR(150) UNIQUE,
    date_of_birth   DATE,
    account_opened  DATE,
    region          VARCHAR(50),
    risk_score      NUMERIC(4,2)
);

-- -----------------------------------------------------------
-- 2. MERCHANTS
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_name   VARCHAR(150),
    category        VARCHAR(80),
    country_code    CHAR(2),
    is_high_risk    BOOLEAN DEFAULT FALSE
);

-- -----------------------------------------------------------
-- 3. TRANSACTIONS
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID REFERENCES customers(customer_id),
    merchant_id         UUID REFERENCES merchants(merchant_id),
    amount_gbp          NUMERIC(12,2),
    original_currency   CHAR(3),
    original_amount     NUMERIC(12,2),
    transaction_ts      TIMESTAMPTZ,
    terminal_type       VARCHAR(20),
    terminal_country    CHAR(2),
    is_declined         BOOLEAN DEFAULT FALSE,
    is_disputed         BOOLEAN DEFAULT FALSE,
    is_fraud            BOOLEAN DEFAULT FALSE,
    fraud_type          VARCHAR(50)
);

-- -----------------------------------------------------------
-- 4. ANALYTICS SCHEMA (for ETL-processed data & views)
-- -----------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS analytics;

-- -----------------------------------------------------------
-- 5. INDEXES for query performance
-- -----------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_txn_customer
    ON transactions (customer_id);

CREATE INDEX IF NOT EXISTS idx_txn_merchant
    ON transactions (merchant_id);

CREATE INDEX IF NOT EXISTS idx_txn_timestamp
    ON transactions (transaction_ts);

CREATE INDEX IF NOT EXISTS idx_txn_fraud
    ON transactions (is_fraud);
