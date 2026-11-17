"""Expected loss = PD x LGD x EAD, the standard credit-loss decomposition.

PD  probability of default        from the model
LGD loss given default            share of the exposure not recovered (assumption)
EAD exposure at default           what the customer owes when they default

For a credit card EAD is more than today's balance, because customers heading
toward default tend to draw down unused credit first:

    EAD = balance + CCF x (limit - balance)

CCF (credit conversion factor) is the share of the undrawn line expected to be
drawn before default. It is an assumption here, set in the app.
"""
import numpy as np
import pandas as pd


def exposure_at_default(balance, credit_limit, ccf: float = 0.5):
    if not 0 <= ccf <= 1:
        raise ValueError("ccf must be between 0 and 1")
    drawn = np.clip(np.asarray(balance, dtype=float), 0, None)  # credit balances are not exposure
    undrawn = np.clip(np.asarray(credit_limit, dtype=float) - drawn, 0, None)
    return drawn + ccf * undrawn


def expected_loss(pd_scores, ead, lgd: float = 0.6):
    if not 0 <= lgd <= 1:
        raise ValueError("lgd must be between 0 and 1")
    return np.asarray(pd_scores, dtype=float) * lgd * np.asarray(ead, dtype=float)


def portfolio_summary(pd_scores, ead, lgd: float = 0.6) -> dict:
    el = expected_loss(pd_scores, ead, lgd)
    total_ead = float(np.sum(ead))
    return {
        "accounts": int(len(el)),
        "exposure_at_default": total_ead,
        "expected_loss": float(el.sum()),
        "expected_loss_rate": float(el.sum() / total_ead) if total_ead else 0.0,
        "average_pd": float(np.mean(pd_scores)),
        # concentration: share of expected loss coming from the riskiest 10% of accounts
        "el_share_top_decile": float(np.sort(el)[::-1][: max(1, len(el) // 10)].sum() / el.sum()) if el.sum() else 0.0,
    }


def realised_loss(y, ead, lgd: float = 0.6) -> float:
    """Loss on the accounts that actually defaulted, under the same LGD and EAD assumptions."""
    return float(np.sum(np.asarray(y) * lgd * np.asarray(ead, dtype=float)))


def loss_by_grade(grades, pd_scores, ead, lgd: float = 0.6) -> pd.DataFrame:
    df = pd.DataFrame({"grade": grades, "pd": pd_scores, "ead": ead, "el": expected_loss(pd_scores, ead, lgd)})
    out = df.groupby("grade").agg(
        accounts=("pd", "size"), average_pd=("pd", "mean"), exposure_at_default=("ead", "sum"), expected_loss=("el", "sum")
    )
    out["share_of_expected_loss"] = out["expected_loss"] / out["expected_loss"].sum()
    return out.reset_index()
