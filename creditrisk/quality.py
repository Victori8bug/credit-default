"""Data-quality controls run before any modelling.

Each check returns pass/fail plus what was found, so a failed load is caught
and explained instead of silently producing a wrong model.
"""
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
TARGET_RAW = "default payment next month"


def run_checks(raw: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add("Row count is 30,000", len(raw) == 30_000, f"{len(raw):,} rows")
    add("Account IDs are unique", raw["ID"].is_unique, f"{raw['ID'].nunique():,} distinct IDs")
    n_null = int(raw.isna().sum().sum())
    add("No missing values", n_null == 0, f"{n_null} missing cells")
    add("Target is 0 or 1", set(raw[TARGET_RAW].unique()) <= {0, 1}, f"values {sorted(int(v) for v in raw[TARGET_RAW].unique())}")
    add("Credit limits are positive", (raw["LIMIT_BAL"] > 0).all(), f"min {raw['LIMIT_BAL'].min():,}")
    lo, hi = int(raw[PAY_COLS].min().min()), int(raw[PAY_COLS].max().max())
    add("Repayment codes within -2..8", lo >= -2 and hi <= 8, f"range {lo}..{hi}")
    neg_pay = int((raw[[f"PAY_AMT{i}" for i in range(1, 7)]] < 0).sum().sum())
    add("Payment amounts are not negative", neg_pay == 0, f"{neg_pay} negative values")
    over = float((raw["BILL_AMT1"] > raw["LIMIT_BAL"]).mean())
    # Informational: over-limit balances are real, but worth knowing about.
    add("Under 15% of balances exceed the limit", over < 0.15, f"{over:.1%} of accounts over limit")
    return pd.DataFrame(checks)
