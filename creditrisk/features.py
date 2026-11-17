"""Turn the raw account history into model inputs a credit analyst would recognise.

Why this exists: fed the raw columns, logistic regression scored a held-out AUC
of only 0.71 and its probabilities were unreliable. The raw repayment codes are
not a straight line of risk (-2 = no balance, -1 = paid in full, 0 = paying the
minimum, 1..8 = months late), and dollar balances are heavily skewed. The
features below fix both problems and lift AUC to about 0.76.
"""
import numpy as np
import pandas as pd

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]

ENGINEERED_FEATURES = [
    "months_late_now",
    "worst_months_late_6m",
    "count_months_late_6m",
    "paid_in_full_now",
    "no_balance_now",
    "utilization",
    "payment_ratio",
    "log_credit_limit",
    "log_last_payment",
]


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    pay = X[PAY_COLS]
    late = pay.clip(lower=0)  # codes below 1 all mean "not late"

    out["months_late_now"] = late["PAY_1"]
    out["worst_months_late_6m"] = late.max(axis=1)
    out["count_months_late_6m"] = (pay > 0).sum(axis=1)
    out["paid_in_full_now"] = (X["PAY_1"] == -1).astype(int)
    out["no_balance_now"] = (X["PAY_1"] == -2).astype(int)

    # Share of the credit line in use. Capped because a few balances exceed the limit.
    out["utilization"] = (X["BILL_AMT1"] / X["LIMIT_BAL"]).clip(0, 1.5)

    # Share of last month's bill that was paid; 1.0 when there was nothing to pay.
    prior_bill = X["BILL_AMT2"].clip(lower=0)
    ratio = (X["PAY_AMT1"] / prior_bill.where(prior_bill > 0)).clip(0, 1)
    out["payment_ratio"] = ratio.fillna(1.0)

    # Logs compress the long right tail of dollar amounts.
    out["log_credit_limit"] = np.log(X["LIMIT_BAL"])
    out["log_last_payment"] = np.log1p(X["PAY_AMT1"])

    # TODO (Victoria): add a feature of your own and see whether test AUC moves.
    # Idea: balance trend = (BILL_AMT1 - BILL_AMT6) / LIMIT_BAL, i.e. is debt growing?
    # Add the column here, append its name to ENGINEERED_FEATURES, rerun pytest and the app.

    return out[ENGINEERED_FEATURES]
