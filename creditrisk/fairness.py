"""Outcome testing: does the approval policy treat demographic groups differently?

Demographics are never model inputs (see data.py). They are used here only to
test outcomes. The adverse impact ratio (AIR) divides each group's approval
rate by the highest group's approval rate; under the "four-fifths" convention a
ratio below 0.80 is flagged for review. A flag is a prompt to investigate, not
proof of discrimination, and a pass is not a legal safe harbour.
"""
import numpy as np
import pandas as pd

SEX = {1: "Male", 2: "Female"}
EDUCATION = {1: "Graduate school", 2: "University", 3: "High school"}  # other codes -> "Other / unknown"
MARRIAGE = {1: "Married", 2: "Single"}                                # other codes -> "Other / unknown"
AGE_BINS = [0, 29, 39, 49, 200]
AGE_LABELS = ["Under 30", "30-39", "40-49", "50 and over"]


def demographics(raw: pd.DataFrame) -> pd.DataFrame:
    """Readable group labels from the raw demographic codes."""
    return pd.DataFrame(
        {
            "Sex": raw["SEX"].map(SEX).fillna("Other / unknown"),
            "Age band": pd.cut(raw["AGE"], AGE_BINS, labels=AGE_LABELS).astype(str),
            "Education": raw["EDUCATION"].map(EDUCATION).fillna("Other / unknown"),
            "Marital status": raw["MARRIAGE"].map(MARRIAGE).fillna("Other / unknown"),
        },
        index=raw.index,
    )


def adverse_impact(groups, approved, y=None, min_group: int = 500) -> pd.DataFrame:
    """Approval rate and adverse impact ratio per group.

    Groups smaller than min_group are reported but neither flagged nor used as the
    reference, because a small group's approval rate is too noisy to anchor the ratio.
    """
    df = pd.DataFrame({"group": np.asarray(groups), "approved": np.asarray(approved).astype(int)})
    agg = {"accounts": ("approved", "size"), "approval_rate": ("approved", "mean")}
    if y is not None:
        df["y"] = np.asarray(y)
        agg["observed_default_rate"] = ("y", "mean")
    out = df.groupby("group").agg(**agg).reset_index()
    reference = out.loc[out["accounts"] >= min_group, "approval_rate"].max()
    out["adverse_impact_ratio"] = out["approval_rate"] / reference if reference else np.nan
    out["flag"] = np.where(
        out["accounts"] < min_group, "Too few to assess",
        np.where(out["adverse_impact_ratio"] < 0.80, "Review", "OK"),
    )
    return out.sort_values("approval_rate", ascending=False).reset_index(drop=True)
