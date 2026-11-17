"""Link the portfolio to current US credit conditions using Federal Reserve data.

Three public series from FRED (Federal Reserve Economic Data, St. Louis Fed):
  CORCCACBS  charge-off rate on credit card loans, all commercial banks (quarterly, annualised %)
  DRCCLACBS  delinquency rate on credit card loans, all commercial banks (quarterly, %)
  UNRATE     US unemployment rate (monthly, %)

The app pulls them live each day. If the pull fails it falls back to the
snapshot saved in data/fred_snapshot.csv, and says so on screen.

Two uses:
1. Re-base: the sample accounts default at 22% (Taiwan, 2005 crisis). We shift
   every PD so the portfolio average matches what today's US charge-off rate
   implies, keeping each account's rank. Identity used:
       charge-off rate ~= PD x LGD   so   PD ~= charge-off rate / LGD
2. Macro stress: a regression of the one-year change in the charge-off rate on
   the one-year change in unemployment tells us what a Fed stress scenario
   (unemployment peaking at 10%) would imply for charge-offs, and so for PD.
"""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = {
    "CORCCACBS": "Card charge-off rate",
    "DRCCLACBS": "Card delinquency rate",
    "UNRATE": "Unemployment rate",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1991-01-01"
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "fred_snapshot.csv"

# Peak unemployment in the Federal Reserve's 2026 supervisory "severely adverse" scenario
# (Board press release, 4 February 2026). Update this each year when the Fed publishes.
FED_SEVERELY_ADVERSE_UNEMPLOYMENT = 10.0

# 2020-2022 is left out of the regression: stimulus payments and forbearance meant
# unemployment hit 14.8% while card charge-offs FELL, the opposite of every other cycle.
EXCLUDE_FROM_FIT = ("2020-01-01", "2022-12-31")


def load_snapshot(path=SNAPSHOT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def fetch_live(timeout: float = 8.0) -> pd.DataFrame:
    """Download the three series from FRED. Raises on any network or format problem."""
    frames = []
    for series_id in SERIES:
        with urllib.request.urlopen(FRED_URL.format(series_id=series_id), timeout=timeout) as resp:
            raw = pd.read_csv(io.BytesIO(resp.read()))
        if raw.shape[1] != 2 or len(raw) < 100:
            raise ValueError(f"Unexpected FRED response for {series_id}")
        raw.columns = ["date", "value"]
        raw["date"] = pd.to_datetime(raw["date"])
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")  # FRED marks gaps with blanks or "."
        raw.insert(0, "series", series_id)
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def load(live: bool = True):
    """Return (data, source). Source is "live" or "snapshot"."""
    if live:
        try:
            return fetch_live(), "live"
        except Exception:
            pass
    return load_snapshot(), "snapshot"


def quarterly_panel(long: pd.DataFrame) -> pd.DataFrame:
    """One row per quarter: unemployment (average of the months), charge-off and delinquency rates."""
    wide = {sid: long[long["series"] == sid].set_index("date")["value"] for sid in SERIES}
    panel = pd.DataFrame(
        {
            "unemployment": wide["UNRATE"].resample("QS").mean(),
            "chargeoff": wide["CORCCACBS"],
            "delinquency": wide["DRCCLACBS"],
        }
    )
    return panel.dropna()


def latest(long: pd.DataFrame) -> dict:
    """Most recent published value and date of each series."""
    out = {}
    for sid in SERIES:
        s = long[long["series"] == sid].dropna(subset=["value"]).sort_values("date")
        out[sid] = {"value": float(s["value"].iloc[-1]), "date": s["date"].iloc[-1]}
    return out


def ols(x, y) -> dict:
    """Simple linear regression y = intercept + slope * x, written out from the formulas."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_bar, y_bar = x.mean(), y.mean()
    slope = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    intercept = y_bar - slope * x_bar
    residuals = y - (intercept + slope * x)
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y_bar) ** 2)
    return {"intercept": float(intercept), "slope": float(slope), "r2": float(r2), "n": int(len(x))}


def yoy_changes(panel: pd.DataFrame, exclude=EXCLUDE_FROM_FIT) -> pd.DataFrame:
    """One-year change in unemployment and in the charge-off rate, quarter by quarter."""
    d = pd.DataFrame(
        {"d_unemployment": panel["unemployment"].diff(4), "d_chargeoff": panel["chargeoff"].diff(4)}
    ).dropna()
    if exclude:
        d = d[(d.index < exclude[0]) | (d.index > exclude[1])]
    return d


def fit_chargeoff_model(panel: pd.DataFrame, exclude=EXCLUDE_FROM_FIT) -> dict:
    """How many points does the charge-off rate move when unemployment moves one point over a year?"""
    d = yoy_changes(panel, exclude)
    return ols(d["d_unemployment"], d["d_chargeoff"])


def scenario_chargeoff(current_chargeoff, current_unemployment, scenario_unemployment, fit) -> float:
    """Charge-off rate implied by a move in unemployment from today's level to the scenario level."""
    change = fit["intercept"] + fit["slope"] * (scenario_unemployment - current_unemployment)
    return max(0.0, current_chargeoff + change)


def implied_pd(chargeoff_pct: float, lgd: float) -> float:
    """Charge-off rate ~= PD x LGD, so PD ~= charge-off rate / LGD (capped below 100%)."""
    return float(min(chargeoff_pct / 100 / lgd, 0.99))


def rebase_pd(pd_scores, target_mean: float) -> np.ndarray:
    """Shift every account's log-odds by the same amount so the average PD equals target_mean.

    Ranking is untouched: the riskiest account stays the riskiest. Only the level moves.
    The shift is found by bisection, because the average of a logistic curve has no closed form.
    """
    if not 0 < target_mean < 1:
        raise ValueError("target_mean must be between 0 and 1")
    p = np.clip(np.asarray(pd_scores, dtype=float), 1e-6, 1 - 1e-6)
    log_odds = np.log(p / (1 - p))
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(log_odds + mid)))).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(log_odds + (lo + hi) / 2)))


def macro_stress_table(pd_scores, ead, lgd: float, long: pd.DataFrame) -> pd.DataFrame:
    """Portfolio expected loss re-based to today's US conditions and under the Fed scenario."""
    panel, now = quarterly_panel(long), latest(long)
    fit = fit_chargeoff_model(panel)
    co_now, u_now = now["CORCCACBS"]["value"], now["UNRATE"]["value"]
    peak = panel["chargeoff"].idxmax()
    cases = [
        ("Current US conditions", u_now, co_now),
        ("Fed 2026 severely adverse", FED_SEVERELY_ADVERSE_UNEMPLOYMENT,
         scenario_chargeoff(co_now, u_now, FED_SEVERELY_ADVERSE_UNEMPLOYMENT, fit)),
        (f"Historical peak ({peak.year} Q{peak.quarter}, actual)", float(panel.loc[peak, "unemployment"]),
         float(panel.loc[peak, "chargeoff"])),
    ]
    ead = np.asarray(ead, dtype=float)
    rows = []
    for name, unemployment, chargeoff in cases:
        target = implied_pd(chargeoff, lgd)
        el = float(np.sum(rebase_pd(pd_scores, target) * lgd * ead))
        rows.append({"scenario": name, "unemployment_pct": unemployment, "chargeoff_rate_pct": chargeoff,
                     "portfolio_pd": target, "expected_loss": el})
    out = pd.DataFrame(rows)
    out["expected_loss_vs_current"] = out["expected_loss"] / out["expected_loss"].iloc[0] - 1
    return out
