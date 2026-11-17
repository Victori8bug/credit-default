import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

from creditrisk.metrics import auc, calibration_table, confusion_counts, ks_statistic, roc_points

# Four accounts, worked by hand in the README:
# defaulters scored 0.35 and 0.80, good payers scored 0.10 and 0.40.
Y = [0, 0, 1, 1]
S = [0.10, 0.40, 0.35, 0.80]


def test_confusion_counts_hand_example():
    # cutoff 0.38 declines the 0.40 (good) and 0.80 (bad) accounts
    assert confusion_counts(Y, S, 0.38) == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_auc_hand_example():
    # 4 defaulter/good pairs; the defaulter outranks the good payer in 3 of them
    assert auc(Y, S) == pytest.approx(0.75)


def test_ks_hand_example():
    # at cutoff 0.80: TPR 0.5, FPR 0.0 -> gap 0.5, the largest on the curve
    assert ks_statistic(Y, S) == pytest.approx(0.5)


def test_auc_and_ks_match_sklearn_on_random_data_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = np.round(rng.random(5000) * 0.6 + y * 0.2, 2)  # rounding creates tied scores
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    fpr, tpr, _ = roc_curve(y, s)
    assert ks_statistic(y, s) == pytest.approx(np.max(tpr - fpr), abs=1e-12)


def test_roc_starts_at_origin_and_ends_at_one():
    fpr, tpr = roc_points(Y, S)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)


def test_calibration_table_bands():
    y = [0, 0, 0, 1, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    t = calibration_table(y, s, bins=2)
    assert list(t["accounts"]) == [4, 4]
    assert list(t["observed_default_rate"]) == [0.25, 0.75]
    assert t["predicted_pd"].tolist() == pytest.approx([0.25, 0.75])
