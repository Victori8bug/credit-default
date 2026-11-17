"""Model-quality metrics, written out in NumPy so each one can be explained.

Convention everywhere: y = 1 means the account defaulted; score = predicted PD.
The tests check these against scikit-learn and against hand calculations.
"""
import numpy as np
import pandas as pd


def confusion_counts(y, pd_scores, cutoff):
    """Decline when PD >= cutoff. Returns counts from the lender's point of view.

    tp = defaulters declined      fn = defaulters approved (credit losses)
    tn = good payers approved     fp = good payers declined (lost business)
    """
    y = np.asarray(y)
    declined = np.asarray(pd_scores) >= cutoff
    return {
        "tp": int(np.sum(declined & (y == 1))),
        "fp": int(np.sum(declined & (y == 0))),
        "tn": int(np.sum(~declined & (y == 0))),
        "fn": int(np.sum(~declined & (y == 1))),
    }


def roc_points(y, pd_scores):
    """True-positive and false-positive rate at every distinct cutoff.

    Sort accounts from riskiest to safest and walk down the list: each step
    declines one more score level and we record the share of defaulters caught
    (TPR) against the share of good payers wrongly declined (FPR).
    """
    y = np.asarray(y)
    s = np.asarray(pd_scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    cum_bad = np.cumsum(y == 1)
    cum_good = np.cumsum(y == 0)
    last_of_tie = np.r_[s[1:] != s[:-1], True]  # one point per distinct score
    tpr = np.r_[0.0, cum_bad[last_of_tie] / cum_bad[-1]]
    fpr = np.r_[0.0, cum_good[last_of_tie] / cum_good[-1]]
    return fpr, tpr


def auc(y, pd_scores):
    """Area under the ROC curve by the trapezoid rule.

    Equivalent reading: the probability that a randomly chosen defaulter gets a
    higher PD than a randomly chosen good payer. 0.5 = coin flip, 1.0 = perfect.
    """
    fpr, tpr = roc_points(y, pd_scores)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def ks_statistic(y, pd_scores):
    """Kolmogorov-Smirnov: the largest gap between TPR and FPR over all cutoffs.

    A standard credit-scoring measure of how far apart the model pushes the
    score distributions of defaulters and good payers.
    """
    fpr, tpr = roc_points(y, pd_scores)
    return float(np.max(tpr - fpr))


def calibration_table(y, pd_scores, bins: int = 10) -> pd.DataFrame:
    """Compare predicted PD with the observed default rate in equal-sized score bands.

    The profit math treats PD as a real probability, so this check matters:
    if a band predicted 10% and 20% actually defaulted, cutoffs will be wrong.
    """
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_scores, dtype=float)})
    df["band"] = pd.qcut(df["pd"].rank(method="first"), bins, labels=False) + 1
    out = df.groupby("band").agg(accounts=("y", "size"), predicted_pd=("pd", "mean"), observed_default_rate=("y", "mean"))
    return out.reset_index()
