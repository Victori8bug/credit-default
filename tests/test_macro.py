"""Checks for the Federal Reserve data link: regression, re-basing and the live/snapshot fallback."""
import io

import numpy as np
import pandas as pd
import pytest

from creditrisk import macro


def test_ols_hand_example():
    # points (1,3) (2,5) (3,7) lie exactly on y = 1 + 2x
    fit = macro.ols([1, 2, 3], [3, 5, 7])
    assert fit["intercept"] == pytest.approx(1) and fit["slope"] == pytest.approx(2)
    assert fit["r2"] == pytest.approx(1) and fit["n"] == 3


def test_ols_matches_numpy_on_noisy_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.5 + 1.2 * x + rng.normal(scale=0.5, size=200)
    fit = macro.ols(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit["slope"] == pytest.approx(slope) and fit["intercept"] == pytest.approx(intercept)
    assert fit["r2"] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)


def test_implied_pd_hand_calc():
    # 3.82% charge-off rate with 60% loss given default -> 0.0382 / 0.60
    assert macro.implied_pd(3.82, 0.60) == pytest.approx(0.063667, abs=1e-6)
    assert macro.implied_pd(90, 0.60) == 0.99  # capped


def test_scenario_chargeoff_hand_calc():
    fit = {"intercept": 0.1, "slope": 1.0}
    # unemployment 4% -> 10% is +6 points: 3.8 + 0.1 + 1.0 x 6 = 9.9
    assert macro.scenario_chargeoff(3.8, 4.0, 10.0, fit) == pytest.approx(9.9)
    assert macro.scenario_chargeoff(1.0, 10.0, 2.0, fit) == 0.0  # never negative


def test_rebase_hits_target_and_keeps_ranking():
    rng = np.random.default_rng(1)
    p = rng.beta(2, 6, size=5000)
    for target in (0.03, 0.064, 0.25, 0.60):
        out = macro.rebase_pd(p, target)
        assert out.mean() == pytest.approx(target, abs=1e-6)
        assert (np.argsort(out) == np.argsort(p)).all()
    assert macro.rebase_pd(p, p.mean()) == pytest.approx(p, abs=1e-6)  # no change needed
    with pytest.raises(ValueError):
        macro.rebase_pd(p, 1.2)


def test_snapshot_matches_values_published_by_fred():
    long = macro.load_snapshot()
    v = long.set_index(["series", "date"])["value"]
    assert v["CORCCACBS", pd.Timestamp("2009-10-01")] == 10.54   # financial-crisis peak
    assert v["DRCCLACBS", pd.Timestamp("2009-04-01")] == 6.77
    assert v["DRCCLACBS", pd.Timestamp("2026-04-01")] == 2.85
    assert v["UNRATE", pd.Timestamp("2020-04-01")] == 14.8
    panel = macro.quarterly_panel(long)
    # 2009 Q4 unemployment: (10.0 + 9.9 + 9.9) / 3
    assert panel.loc["2009-10-01", "unemployment"] == pytest.approx(9.9333, abs=1e-3)


def test_fit_on_snapshot_is_sensible():
    panel = macro.quarterly_panel(macro.load_snapshot())
    fit = macro.fit_chargeoff_model(panel)
    assert 0.8 < fit["slope"] < 1.3      # about one point of charge-offs per point of unemployment
    assert fit["r2"] > 0.4
    with_covid = macro.fit_chargeoff_model(panel, exclude=None)
    assert with_covid["r2"] < fit["r2"]  # the pandemic years weaken the relationship


def test_falls_back_to_snapshot_when_fred_is_unreachable(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(macro.urllib.request, "urlopen", boom)
    long, source = macro.load(live=True)
    assert source == "snapshot" and len(long) > 500


def test_live_fetch_parses_fred_csv(monkeypatch):
    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=None):
        sid = url.split("id=")[1].split("&")[0]
        rows = "\n".join(f"{d.date()},{1 + i * 0.01:.2f}" for i, d in enumerate(pd.date_range("1991-01-01", periods=150, freq="QS")))
        return FakeResponse(f"observation_date,{sid}\n{rows}\n2028-07-01,\n".encode())

    monkeypatch.setattr(macro.urllib.request, "urlopen", fake_urlopen)
    long, source = macro.load(live=True)
    assert source == "live" and set(long["series"]) == set(macro.SERIES)
    assert macro.latest(long)["UNRATE"]["value"] == pytest.approx(2.49)  # the trailing blank row is skipped


def test_macro_stress_table_orders_scenarios():
    rng = np.random.default_rng(2)
    p, ead = rng.beta(2, 6, size=2000), rng.uniform(1_000, 50_000, size=2000)
    t = macro.macro_stress_table(p, ead, 0.6, macro.load_snapshot())
    assert t["expected_loss"].iloc[1] > t["expected_loss"].iloc[0]
    assert t["portfolio_pd"].iloc[0] == pytest.approx(macro.implied_pd(t["chargeoff_rate_pct"].iloc[0], 0.6))
