import numpy as np
import pytest

from creditrisk.threshold import best_cutoff, breakeven_pd, portfolio_outcomes, sweep


def test_breakeven_pd_hand_calc():
    # 10% margin, 60% loss given default -> 0.10 / 0.70
    assert breakeven_pd(0.10, 0.60) == pytest.approx(1 / 7)
    # at the break-even PD the expected profit per unit lent is zero
    p = breakeven_pd(0.10, 0.60)
    assert (1 - p) * 0.10 - p * 0.60 == pytest.approx(0.0)


def test_breakeven_rejects_bad_inputs():
    with pytest.raises(ValueError):
        breakeven_pd(0.10, 0.0)


def test_portfolio_outcomes_hand_example():
    # 5 applicants, PDs 0.05 0.10 0.20 0.30 0.50; the 0.20 and 0.50 accounts default.
    y = [0, 0, 1, 0, 1]
    s = [0.05, 0.10, 0.20, 0.30, 0.50]
    out = portfolio_outcomes(y, s, cutoff=0.25, loan_amount=10_000, margin=0.10, lgd=0.60)
    # approved: 0.05, 0.10, 0.20 -> 2 good, 1 bad
    assert out["approval_rate"] == pytest.approx(0.6)
    assert out["bad_rate_approved"] == pytest.approx(1 / 3)
    assert out["defaults_avoided"] == pytest.approx(0.5)
    # 2 x 1,000 earned - 1 x 6,000 lost
    assert out["profit"] == pytest.approx(-4_000)
    assert out["profit_per_applicant"] == pytest.approx(-800)


def test_approving_nobody_earns_zero():
    out = portfolio_outcomes([0, 1], [0.2, 0.9], cutoff=0.0)
    assert out["approval_rate"] == 0 and out["profit"] == 0 and out["bad_rate_approved"] == 0


def test_best_cutoff_lands_near_breakeven_when_pd_is_calibrated():
    # Simulate applicants whose true default probability equals the model PD.
    rng = np.random.default_rng(1)
    pd_scores = rng.random(200_000)
    y = (rng.random(200_000) < pd_scores).astype(int)
    df = sweep(y, pd_scores, margin=0.10, lgd=0.60)
    assert best_cutoff(df) == pytest.approx(breakeven_pd(0.10, 0.60), abs=0.03)


def test_approval_rate_rises_with_cutoff():
    rng = np.random.default_rng(2)
    s = rng.random(1000)
    y = (rng.random(1000) < s).astype(int)
    df = sweep(y, s)
    assert df["approval_rate"].is_monotonic_increasing
