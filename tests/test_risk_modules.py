"""Hand-worked checks for the scorecard, loss, stress, monitoring, fairness and reason-code modules."""
import numpy as np
import pandas as pd
import pytest

from creditrisk import data, explain, fairness, features, loss, model, monitoring, quality, scorecard, stress, warehouse


# ---------------------------------------------------------------- scorecard
def test_score_scaling_hand_calc():
    # 50:1 odds of repaying means PD = 1/51 and must score exactly 600
    assert scorecard.pd_to_score(1 / 51) == pytest.approx(600)
    # doubling the odds to 100:1 adds exactly 20 points
    assert scorecard.pd_to_score(1 / 101) == pytest.approx(620)
    # textbook constants: factor 28.8539, offset 487.123
    factor, offset = scorecard._factor_offset()
    assert factor == pytest.approx(28.8539, abs=1e-4)
    assert offset == pytest.approx(487.123, abs=1e-3)


def test_score_round_trip_and_grades():
    p = np.array([0.02, 0.07, 0.15, 0.30, 0.60])
    assert scorecard.score_to_pd(scorecard.pd_to_score(p)) == pytest.approx(p)
    assert list(scorecard.risk_grade(p)) == ["A", "B", "C", "D", "E"]
    assert scorecard.risk_grade([0.05])[0] == "B"  # boundary goes to the riskier grade


# --------------------------------------------------------------------- loss
def test_ead_and_expected_loss_hand_calc():
    # balance 40,000 on a 100,000 line, half the unused 60,000 expected to be drawn
    ead = loss.exposure_at_default([40_000], [100_000], ccf=0.5)
    assert ead[0] == pytest.approx(70_000)
    # PD 10% x LGD 60% x EAD 70,000
    assert loss.expected_loss([0.10], ead, lgd=0.6)[0] == pytest.approx(4_200)


def test_ead_edge_cases():
    # credit balance counts as zero drawn; over-limit balance has nothing undrawn
    assert loss.exposure_at_default([-500], [10_000], 0.5)[0] == pytest.approx(5_000)
    assert loss.exposure_at_default([12_000], [10_000], 0.5)[0] == pytest.approx(12_000)
    with pytest.raises(ValueError):
        loss.exposure_at_default([1], [1], ccf=1.5)


def test_portfolio_summary_adds_up():
    s = loss.portfolio_summary([0.1, 0.5], [1_000, 1_000], lgd=0.5)
    assert s["expected_loss"] == pytest.approx(300)       # 50 + 250
    assert s["expected_loss_rate"] == pytest.approx(0.15)


# --------------------------------------------------------------- monitoring
def test_psi_hand_calc():
    # two bins shift from 50/50 to 60/40:
    # (0.6-0.5)ln(0.6/0.5) + (0.4-0.5)ln(0.4/0.5) = 0.018232 + 0.022314
    assert monitoring.psi_from_shares([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.040546, abs=1e-6)
    assert monitoring.psi_from_shares([0.5, 0.5], [0.5, 0.5]) == 0


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 20_000)
    assert monitoring.psi(base, rng.normal(0, 1, 20_000)) < 0.01
    assert monitoring.psi(base, rng.normal(1, 1, 20_000)) > 0.25
    assert monitoring.band(0.05) == "Stable" and monitoring.band(0.2) == "Investigate"
    assert monitoring.band(0.3) == "Material shift"


# ----------------------------------------------------------------- fairness
def test_adverse_impact_ratio_hand_calc():
    # group X: 50 of 100 approved; group Y: 40 of 100 approved -> ratio 0.40 / 0.50 = 0.80
    groups = ["X"] * 100 + ["Y"] * 100
    approved = [1] * 50 + [0] * 50 + [1] * 40 + [0] * 60
    t = fairness.adverse_impact(groups, approved, min_group=50).set_index("group")
    assert t.loc["Y", "adverse_impact_ratio"] == pytest.approx(0.80)
    assert t.loc["Y", "flag"] == "OK"          # exactly four-fifths is not below it
    approved[100] = 0                           # one fewer approval: 39% -> ratio 0.78
    t = fairness.adverse_impact(groups, approved, min_group=50).set_index("group")
    assert t.loc["Y", "flag"] == "Review"


def test_small_groups_are_not_flagged():
    t = fairness.adverse_impact(["X"] * 600 + ["Z"] * 5, [1] * 600 + [0] * 5).set_index("group")
    assert t.loc["Z", "flag"] == "Too few to assess"


# -------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def fitted():
    df = data.load_clean()
    X_tr, X_te, y_tr, y_te = model.split(df)
    return model.build_logit().fit(X_tr, y_tr), X_te, y_te


def test_reason_code_contributions_rebuild_the_pd(fitted):
    m, X_te, _ = fitted
    contrib = explain.contributions(m, X_te.head(50))
    rebuilt = [explain.pd_from_contributions(m, row) for _, row in contrib.iterrows()]
    assert rebuilt == pytest.approx(model.predict_pd(m, X_te.head(50)), abs=1e-9)


def test_reason_codes_are_capped_and_readable(fitted):
    m, X_te, _ = fitted
    reasons = explain.reason_codes(m, X_te.head(200), top_n=4)
    assert all(len(r) <= 4 for r in reasons)
    assert all(text in explain.REASON_TEXT.values() for r in reasons for text in r)


def test_stress_scenarios_raise_risk_in_order(fitted):
    m, X_te, _ = fitted
    avg = [model.predict_pd(m, stress.apply(X_te, s)).mean() for s in stress.SCENARIOS]
    assert avg[0] < avg[1] < avg[2]  # baseline < adverse < severe


def test_baseline_scenario_changes_nothing(fitted):
    _, X_te, _ = fitted
    pd.testing.assert_frame_equal(stress.apply(X_te, stress.SCENARIOS[0]), X_te, check_dtype=False)


def test_adverse_scenario_shock_hand_example():
    row = {c: 0 for c in data.FEATURES}
    row.update({"LIMIT_BAL": 100_000, "PAY_1": 2, "BILL_AMT1": 10_000, "BILL_AMT2": -200, "PAY_AMT1": 1_000})
    out = stress.apply(pd.DataFrame([row]), stress.SCENARIOS[1]).iloc[0]
    assert out["PAY_AMT1"] == pytest.approx(750)      # payments x 0.75
    assert out["BILL_AMT1"] == pytest.approx(11_000)  # balances x 1.10
    assert out["BILL_AMT2"] == -200                   # credit balances untouched
    assert out["PAY_1"] == 3                          # already late -> one more month


def test_credit_line_cut_scenario_hand_example():
    row = {c: 0 for c in data.FEATURES}
    row.update({"LIMIT_BAL": 100_000, "BILL_AMT1": 40_000, "PAY_AMT1": 1_000})
    cut = next(s for s in stress.SCENARIOS if s.name == "Credit line cut")
    out = stress.apply(pd.DataFrame([row]), cut).iloc[0]
    assert out["LIMIT_BAL"] == pytest.approx(80_000)   # limit x 0.80
    assert out["BILL_AMT1"] == 40_000 and out["PAY_AMT1"] == 1_000  # customer unchanged
    # exposure falls: 40,000 + 0.5 x 40,000 unused = 60,000, down from 70,000
    assert loss.exposure_at_default([out["BILL_AMT1"]], [out["LIMIT_BAL"]], 0.5)[0] == pytest.approx(60_000)


def test_sql_features_match_pandas_features():
    con = warehouse.connect()
    sql = con.execute("SELECT * FROM account_features ORDER BY account_id").df().drop(columns="account_id")
    pandas_version = features.engineer(data.load_clean()[data.FEATURES]).reset_index(drop=True)
    for col in features.ENGINEERED_FEATURES:
        assert np.allclose(sql[col].to_numpy(dtype=float), pandas_version[col].to_numpy(dtype=float)), col


def test_portfolio_query_covers_every_account():
    con = warehouse.connect()
    t = warehouse.query(con, "03_portfolio_by_delinquency.sql")
    assert int(t["accounts"].sum()) == 30_000
    assert t["default_rate_pct"].is_monotonic_increasing  # later delinquency -> higher default rate


def test_data_quality_checks_pass_on_shipped_file():
    checks = quality.run_checks(data.load_raw())
    assert checks["passed"].all(), checks[~checks["passed"]]
