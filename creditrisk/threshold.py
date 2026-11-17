"""Approval-threshold economics: turn a PD into an approve/decline decision.

Decision rule: approve when PD < cutoff.

Profit model (deliberately simple, every input is an assumption set in the app):
  good loan  -> lender earns   margin x loan_amount
  bad loan   -> lender loses   lgd    x loan_amount   (lgd = loss given default)
  declined   -> 0

Expected profit of one loan with default probability p:
  (1 - p) x margin - p x lgd      (per unit lent)
This is positive only when p < margin / (margin + lgd), the break-even PD.
"""
import numpy as np
import pandas as pd


def breakeven_pd(margin: float, lgd: float) -> float:
    """PD at which a loan's expected profit is exactly zero."""
    if margin < 0 or lgd <= 0:
        raise ValueError("margin must be >= 0 and lgd must be > 0")
    return margin / (margin + lgd)


def portfolio_outcomes(y, pd_scores, cutoff, loan_amount=10_000.0, margin=0.10, lgd=0.60) -> dict:
    """Realised results on a set of accounts if we approve everything with PD < cutoff."""
    y = np.asarray(y)
    approved = np.asarray(pd_scores) < cutoff
    n = len(y)
    n_approved = int(approved.sum())
    goods_approved = int(np.sum(approved & (y == 0)))
    bads_approved = int(np.sum(approved & (y == 1)))
    total_bads = int(np.sum(y == 1))
    profit = loan_amount * (margin * goods_approved - lgd * bads_approved)
    return {
        "cutoff": float(cutoff),
        "approval_rate": n_approved / n,
        # share of approved accounts that went on to default
        "bad_rate_approved": bads_approved / n_approved if n_approved else 0.0,
        # share of all defaulters the cutoff screened out
        "defaults_avoided": (total_bads - bads_approved) / total_bads if total_bads else 0.0,
        "goods_approved": goods_approved,
        "bads_approved": bads_approved,
        "profit": float(profit),
        "profit_per_applicant": float(profit / n),
    }


def sweep(y, pd_scores, loan_amount=10_000.0, margin=0.10, lgd=0.60, cutoffs=None) -> pd.DataFrame:
    """Outcomes at every cutoff from 1% to 100%: the full trade-off curve."""
    if cutoffs is None:
        cutoffs = np.round(np.arange(0.01, 1.0001, 0.01), 2)
    rows = [portfolio_outcomes(y, pd_scores, c, loan_amount, margin, lgd) for c in cutoffs]
    return pd.DataFrame(rows)


def best_cutoff(sweep_df: pd.DataFrame) -> float:
    """Cutoff with the highest realised profit on the data supplied."""
    return float(sweep_df.loc[sweep_df["profit"].idxmax(), "cutoff"])
