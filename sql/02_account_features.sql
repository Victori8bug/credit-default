-- The nine model inputs, written in SQL. A test checks that this view matches
-- the pandas version in creditrisk/features.py row for row.
CREATE OR REPLACE VIEW account_features AS
SELECT
    account_id,
    GREATEST(PAY_1, 0)                                              AS months_late_now,
    GREATEST(PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, 0)           AS worst_months_late_6m,
    (PAY_1 > 0)::INT + (PAY_2 > 0)::INT + (PAY_3 > 0)::INT
      + (PAY_4 > 0)::INT + (PAY_5 > 0)::INT + (PAY_6 > 0)::INT      AS count_months_late_6m,
    (PAY_1 = -1)::INT                                               AS paid_in_full_now,
    (PAY_1 = -2)::INT                                               AS no_balance_now,
    LEAST(GREATEST(BILL_AMT1 * 1.0 / credit_limit, 0), 1.5)         AS utilization,
    CASE WHEN BILL_AMT2 > 0
         THEN LEAST(GREATEST(PAY_AMT1 * 1.0 / BILL_AMT2, 0), 1)
         ELSE 1.0 END                                               AS payment_ratio,
    LN(credit_limit)                                                AS log_credit_limit,
    LN(1 + PAY_AMT1)                                                AS log_last_payment
FROM accounts;
