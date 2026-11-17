"""Convert a PD into a credit score and a risk grade.

Banks rarely show raw probabilities. They rescale the log-odds of repaying so
that a fixed number of points doubles the odds:

    score = offset + factor x ln(odds of repaying),   odds = (1 - PD) / PD
    factor = PDO / ln(2)
    offset = base_score - factor x ln(base_odds)

Defaults follow the common textbook convention: 600 points at 50:1 odds,
20 points to double the odds (PDO). Higher score = lower risk.
"""
import numpy as np
import pandas as pd

BASE_SCORE, BASE_ODDS, PDO = 600.0, 50.0, 20.0

# PD bands for risk grades (upper bounds). An assumption, not an industry standard.
GRADE_BOUNDS = [0.05, 0.10, 0.20, 0.40]
GRADE_LABELS = ["A", "B", "C", "D", "E"]


def _factor_offset(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    factor = pdo / np.log(2)
    return factor, base_score - factor * np.log(base_odds)


def pd_to_score(pd_scores, base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    p = np.clip(np.asarray(pd_scores, dtype=float), 1e-6, 1 - 1e-6)
    factor, offset = _factor_offset(base_score, base_odds, pdo)
    return offset + factor * np.log((1 - p) / p)


def score_to_pd(score, base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    factor, offset = _factor_offset(base_score, base_odds, pdo)
    odds = np.exp((np.asarray(score, dtype=float) - offset) / factor)
    return 1 / (1 + odds)


def risk_grade(pd_scores) -> np.ndarray:
    """A (PD under 5%) to E (PD 40% or more)."""
    idx = np.searchsorted(GRADE_BOUNDS, np.asarray(pd_scores, dtype=float), side="right")
    return np.array(GRADE_LABELS)[idx]


def grade_table(pd_scores, y=None, ead=None) -> pd.DataFrame:
    df = pd.DataFrame({"grade": risk_grade(pd_scores), "pd": np.asarray(pd_scores, dtype=float)})
    agg = {"accounts": ("pd", "size"), "average_pd": ("pd", "mean")}
    if y is not None:
        df["y"] = np.asarray(y)
        agg["observed_default_rate"] = ("y", "mean")
    if ead is not None:
        df["ead"] = np.asarray(ead, dtype=float)
        agg["exposure_at_default"] = ("ead", "sum")
    return df.groupby("grade").agg(**agg).reindex(GRADE_LABELS).dropna(how="all").reset_index()
