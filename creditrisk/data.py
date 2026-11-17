"""Load and clean the UCI "Default of Credit Card Clients" dataset.

Source: Yeh, I-C. (2009), UCI Machine Learning Repository, dataset #350.
30,000 Taiwanese credit card accounts observed April-September 2005.
Target = 1 if the account defaulted on its payment in October 2005.
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "credit_default.csv"

TARGET = "default"

# Demographic columns are deliberately NOT used as model inputs.
# US fair-lending law (ECOA / Regulation B) bars credit decisions based on sex,
# marital status and (with narrow exceptions) age. Education is not a protected
# class but can act as a proxy for one, so it is excluded too.
EXCLUDED_DEMOGRAPHICS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

# PAY_n  = repayment status n months back (-2/-1/0 = paid or no balance, 1..8 = months late)
# BILL_AMTn = statement balance, PAY_AMTn = amount paid (New Taiwan dollars)
FEATURES = (
    ["LIMIT_BAL"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with the 19 model features plus the 0/1 target column."""
    df = raw.rename(
        columns={
            "default payment next month": TARGET,
            "default.payment.next.month": TARGET,
            "PAY_0": "PAY_1",  # the source file skips PAY_1; rename for consistency
        }
    )
    df = df.drop(columns=["ID"] + EXCLUDED_DEMOGRAPHICS, errors="ignore")
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values; expected none.")
    return df[FEATURES + [TARGET]].copy()


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
