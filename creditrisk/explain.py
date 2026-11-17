"""Reason codes: why did the model score this account the way it did?

Lenders must give declined customers the specific principal reasons for the
decision. With logistic regression this is exact arithmetic, not an
approximation: the log-odds of default is

    intercept + sum of (coefficient x standardised feature value)

so each term is that feature's push above or below the average account.
The largest positive terms are the reasons.
"""
import numpy as np
import pandas as pd

from .features import ENGINEERED_FEATURES

REASON_TEXT = {
    "months_late_now": "Most recent payment is past due",
    "worst_months_late_6m": "Serious delinquency in the last six months",
    "count_months_late_6m": "Number of months with a late payment in the last six",
    "utilization": "High balance relative to credit limit",
    "payment_ratio": "Low payment relative to the previous statement balance",
    "log_credit_limit": "Low credit limit",
    "log_last_payment": "Small most-recent payment amount",
    "paid_in_full_now": "Account status this month: paid in full (see validation finding)",
    "no_balance_now": "Account status this month: no balance (see validation finding)",
}


def contributions(logit_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Per-account, per-feature contribution to the log-odds of default."""
    engineered = logit_pipeline[0].transform(X)
    z = logit_pipeline.named_steps["standardscaler"].transform(engineered)
    coef = logit_pipeline.named_steps["logisticregression"].coef_[0]
    return pd.DataFrame(z * coef, columns=ENGINEERED_FEATURES, index=X.index)


def intercept(logit_pipeline) -> float:
    return float(logit_pipeline.named_steps["logisticregression"].intercept_[0])


def reason_codes(logit_pipeline, X: pd.DataFrame, top_n: int = 4) -> list[list[str]]:
    """Up to top_n reasons per account, strongest first. Only factors that raised the risk count."""
    contrib = contributions(logit_pipeline, X)
    out = []
    for _, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_n)
        out.append([REASON_TEXT[f] for f in top.index])
    return out


def pd_from_contributions(logit_pipeline, contrib_row) -> float:
    """Rebuild the PD from the parts: proves the contributions add up to the model's answer."""
    log_odds = intercept(logit_pipeline) + float(np.sum(contrib_row))
    return float(1 / (1 + np.exp(-log_odds)))
