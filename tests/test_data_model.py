import pytest

import pandas as pd

from creditrisk import data, features, metrics, model


@pytest.fixture(scope="module")
def df():
    return data.load_clean()


def test_dataset_matches_published_description(df):
    # UCI documents 30,000 accounts, 6,636 of which defaulted (22.12%)
    assert len(df) == 30_000
    assert int(df[data.TARGET].sum()) == 6_636


def test_no_demographic_columns_reach_the_model(df):
    assert not set(data.EXCLUDED_DEMOGRAPHICS) & set(df.columns)
    assert len(data.FEATURES) == 19


def test_split_is_stratified_and_reproducible(df):
    X_tr, X_te, y_tr, y_te = model.split(df)
    assert len(X_te) == 9_000
    assert y_tr.mean() == pytest.approx(y_te.mean(), abs=0.001)
    assert model.split(df)[1].index.equals(X_te.index)


def test_logit_beats_chance_on_held_out_accounts(df):
    X_tr, X_te, y_tr, y_te = model.split(df)
    m = model.build_logit().fit(X_tr, y_tr)
    assert metrics.auc(y_te, model.predict_pd(m, X_te)) > 0.74


def test_engineered_features_hand_example():
    # One account: 2 months late now, late in 3 of 6 months, worst was 3 months.
    row = {"LIMIT_BAL": 100_000, "PAY_1": 2, "PAY_2": 3, "PAY_3": 1, "PAY_4": 0, "PAY_5": -1, "PAY_6": -2,
           "BILL_AMT1": 40_000, "BILL_AMT2": 50_000, "PAY_AMT1": 5_000}
    for i in range(3, 7):
        row[f"BILL_AMT{i}"] = 0
    for i in range(2, 7):
        row[f"PAY_AMT{i}"] = 0
    f = features.engineer(pd.DataFrame([row])).iloc[0]
    assert f["months_late_now"] == 2
    assert f["worst_months_late_6m"] == 3
    assert f["count_months_late_6m"] == 3
    assert f["paid_in_full_now"] == 0 and f["no_balance_now"] == 0
    assert f["utilization"] == pytest.approx(0.40)      # 40,000 / 100,000
    assert f["payment_ratio"] == pytest.approx(0.10)    # 5,000 / 50,000


def test_payment_ratio_is_one_when_nothing_was_owed():
    row = {c: 0 for c in data.FEATURES}
    row["LIMIT_BAL"] = 50_000
    assert features.engineer(pd.DataFrame([row])).iloc[0]["payment_ratio"] == 1.0


def test_candidate_balance_trend_hand_example():
    # balance rose from 10,000 to 40,000 on a 100,000 line -> +0.30
    row = {c: 0 for c in data.FEATURES}
    row.update({"LIMIT_BAL": 100_000, "BILL_AMT1": 40_000, "BILL_AMT6": 10_000})
    assert features.balance_trend(pd.DataFrame([row])).iloc[0] == pytest.approx(0.30)
    assert "balance_trend" not in features.ENGINEERED_FEATURES  # tested and rejected
