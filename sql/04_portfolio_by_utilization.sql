-- Default rate by how much of the credit line is in use.
SELECT
    CASE WHEN f.utilization < 0.25 THEN '1: under 25%'
         WHEN f.utilization < 0.50 THEN '2: 25-50%'
         WHEN f.utilization < 0.75 THEN '3: 50-75%'
         WHEN f.utilization < 1.00 THEN '4: 75-100%'
         ELSE '5: over limit' END                       AS utilization_band,
    COUNT(*)                                            AS accounts,
    SUM(a.credit_limit)                                 AS total_credit_limit,
    SUM(GREATEST(a.BILL_AMT1, 0))                       AS balance_outstanding,
    ROUND(100.0 * AVG(a.default_flag), 1)               AS default_rate_pct
FROM accounts a
JOIN account_features f USING (account_id)
GROUP BY 1
ORDER BY 1;
