-- Portfolio view a credit risk analyst would pull first:
-- how do balances and realised default rates line up with current delinquency?
SELECT
    CASE WHEN f.months_late_now = 0 THEN '0 - current'
         WHEN f.months_late_now = 1 THEN '1 month late'
         WHEN f.months_late_now = 2 THEN '2 months late'
         ELSE '3+ months late' END                      AS delinquency_bucket,
    COUNT(*)                                            AS accounts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)  AS pct_of_accounts,
    SUM(GREATEST(a.BILL_AMT1, 0))                       AS balance_outstanding,
    ROUND(100.0 * AVG(a.default_flag), 1)               AS default_rate_pct
FROM accounts a
JOIN account_features f USING (account_id)
GROUP BY 1
ORDER BY 1;
