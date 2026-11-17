"""Ongoing monitoring: has the population the model sees drifted from the one it was built on?

Population Stability Index (PSI) compares two distributions over the same bins:

    PSI = sum over bins of (actual% - expected%) x ln(actual% / expected%)

Common rule of thumb: under 0.10 stable, 0.10 to 0.25 worth investigating,
above 0.25 a material shift. These bands are conventions, not statistical tests.
"""
import numpy as np
import pandas as pd

EPS = 1e-4  # floor so an empty bin does not produce ln(0)


def psi_from_shares(expected_share, actual_share) -> float:
    e = np.clip(np.asarray(expected_share, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual_share, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected, actual, bins: int = 10) -> float:
    """PSI for a numeric variable. Bin edges are deciles of the expected (development) sample."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # nearly constant variable: fall back to its distinct values
        edges = np.unique(np.r_[expected, actual])
        edges = np.r_[edges, edges[-1] + 1]
    edges[0], edges[-1] = -np.inf, np.inf
    e_share = np.histogram(expected, edges)[0] / len(expected)
    a_share = np.histogram(actual, edges)[0] / len(actual)
    return psi_from_shares(e_share, a_share)


def band(value: float) -> str:
    return "Stable" if value < 0.10 else ("Investigate" if value < 0.25 else "Material shift")


def feature_psi(expected_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"variable": c, "psi": psi(expected_df[c], actual_df[c])} for c in expected_df.columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(band)
    return out
