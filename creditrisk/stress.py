"""Stress scenarios: shock customer behaviour, re-score, and re-measure loss.

The dataset has no economic variables, so these are sensitivity scenarios on
the model's inputs, not forecasts tied to unemployment or rates. Each scenario
answers: "if customers paid less, borrowed more and slipped further behind,
how much would expected loss rise, and would the approval cutoff still hold?"
"""
from dataclasses import dataclass

import pandas as pd

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


@dataclass(frozen=True)
class Scenario:
    name: str
    payment_multiplier: float   # customers pay this share of what they paid before
    balance_multiplier: float   # balances grow by this factor
    slip: str                   # who falls one more month behind: "none", "late" or "revolving"
    description: str
    limit_multiplier: float = 1.0  # the lender changes every credit line by this factor


SCENARIOS = [
    Scenario("Baseline", 1.00, 1.00, "none", "Accounts as observed."),
    Scenario("Adverse", 0.75, 1.10, "late",
             "Payments fall 25%, balances rise 10%, accounts already late slip one more month."),
    Scenario("Severe", 0.50, 1.25, "revolving",
             "Payments fall 50%, balances rise 25%, every account carrying a balance slips one month."),
    # A lender action, not a customer shock: what if we cut every credit line by 20%
    # to reduce exposure? Customers behave exactly as before.
    Scenario("Credit line cut", 1.00, 1.00, "none",
             "The lender cuts every credit limit by 20%; customer payments and balances are unchanged.",
             limit_multiplier=0.80),
]


def apply(X: pd.DataFrame, s: Scenario) -> pd.DataFrame:
    """Return a shocked copy of the raw account columns."""
    out = X.copy()
    out["LIMIT_BAL"] = out["LIMIT_BAL"] * s.limit_multiplier
    out[PAY_AMT_COLS] = out[PAY_AMT_COLS] * s.payment_multiplier
    # only positive balances grow; credit balances are left alone
    out[BILL_COLS] = out[BILL_COLS].where(out[BILL_COLS] <= 0, out[BILL_COLS] * s.balance_multiplier)
    if s.slip == "late":
        mask = out["PAY_1"] >= 1
    elif s.slip == "revolving":
        mask = out["PAY_1"] >= 0
    else:
        mask = pd.Series(False, index=out.index)
    out.loc[mask, "PAY_1"] = (out.loc[mask, "PAY_1"] + 1).clip(upper=8)
    return out
