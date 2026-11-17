-- Load the raw CSV into a typed table. {csv_path} is filled in by warehouse.py.
CREATE OR REPLACE TABLE accounts AS
SELECT
    ID                               AS account_id,
    LIMIT_BAL                        AS credit_limit,
    SEX, EDUCATION, MARRIAGE, AGE,
    PAY_0 AS PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,
    "default payment next month"     AS default_flag
FROM read_csv_auto('{csv_path}', header = true);
