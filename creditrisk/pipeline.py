"""One place that runs the whole analysis, shared by the app, the exports and the report."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data, fairness, loss, model, scorecard, stress


@dataclass
class Results:
    raw: pd.DataFrame            # all 30,000 raw rows (includes demographics)
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    fitted: dict                 # model name -> fitted pipeline
    scores: dict                 # model name -> PD on the test accounts
    train_scores: dict           # model name -> PD on the training accounts
    demographics: pd.DataFrame   # group labels for the test accounts


def run() -> Results:
    raw = data.load_raw()
    df = data.clean(raw)
    X_train, X_test, y_train, y_test = model.split(df)
    fitted, scores, train_scores = {}, {}, {}
    for name, build in model.MODELS.items():
        fitted[name] = build().fit(X_train, y_train)
        scores[name] = model.predict_pd(fitted[name], X_test)
        train_scores[name] = model.predict_pd(fitted[name], X_train)
    demo = fairness.demographics(raw).loc[X_test.index]
    return Results(raw, X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), fitted, scores, train_scores, demo)


def stress_table(fitted_model, X: pd.DataFrame, lgd: float, ccf: float, cutoff: float) -> pd.DataFrame:
    """Portfolio PD, exposure and expected loss under each scenario, plus how the approval policy reacts."""
    rows = []
    for s in stress.SCENARIOS:
        Xs = stress.apply(X, s)
        pds = model.predict_pd(fitted_model, Xs)
        ead = loss.exposure_at_default(Xs["BILL_AMT1"], Xs["LIMIT_BAL"], ccf)
        summ = loss.portfolio_summary(pds, ead, lgd)
        rows.append(
            {
                "scenario": s.name,
                "average_pd": summ["average_pd"],
                "exposure_at_default": summ["exposure_at_default"],
                "expected_loss": summ["expected_loss"],
                "expected_loss_rate": summ["expected_loss_rate"],
                "approval_rate_at_cutoff": float(np.mean(pds < cutoff)),
                "share_grade_D_or_E": float(np.mean(np.isin(scorecard.risk_grade(pds), ["D", "E"]))),
            }
        )
    out = pd.DataFrame(rows)
    out["expected_loss_vs_baseline"] = out["expected_loss"] / out["expected_loss"].iloc[0] - 1
    return out


def scored_accounts(res: Results, model_name: str, cutoff: float, lgd: float, ccf: float) -> pd.DataFrame:
    """One row per held-out account: the extract a BI tool (Tableau, Power BI) would connect to."""
    X, pds = res.X_test, res.scores[model_name]
    ead = loss.exposure_at_default(X["BILL_AMT1"], X["LIMIT_BAL"], ccf)
    out = pd.DataFrame(
        {
            "account_id": res.raw.loc[X.index, "ID"].to_numpy(),
            "credit_limit": X["LIMIT_BAL"].to_numpy(),
            "balance": X["BILL_AMT1"].clip(lower=0).to_numpy(),
            "months_late_now": X["PAY_1"].clip(lower=0).to_numpy(),
            "predicted_pd": np.round(pds, 4),
            "credit_score": np.round(scorecard.pd_to_score(pds)).astype(int),
            "risk_grade": scorecard.risk_grade(pds),
            "decision": np.where(pds < cutoff, "Approve", "Decline"),
            "exposure_at_default": np.round(ead, 0),
            "expected_loss": np.round(loss.expected_loss(pds, ead, lgd), 0),
            "actual_default": res.y_test,
        }
    )
    demo = res.demographics.reset_index(drop=True)
    demo.columns = [c.lower().replace(" ", "_") for c in demo.columns]
    return pd.concat([out, demo], axis=1)
