"""Does a candidate feature earn its place? Compare held-out AUC with and without it.

Run from the project folder:  python scripts/evaluate_candidate_feature.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from creditrisk import data, features, metrics, model  # noqa: E402


def main():
    X_train, X_test, y_train, y_test = model.split(data.load_clean())

    def fit_auc(build_inputs):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(build_inputs(X_train), y_train)
        return metrics.auc(y_test, m.predict_proba(build_inputs(X_test))[:, 1]), m[-1].coef_[0]

    def with_candidate(X):
        return pd.concat([features.engineer(X), features.balance_trend(X)], axis=1)

    base_auc, _ = fit_auc(features.engineer)
    cand_auc, coefs = fit_auc(with_candidate)
    print(f"Held-out AUC without balance_trend: {base_auc:.4f}")
    print(f"Held-out AUC with balance_trend:    {cand_auc:.4f}  (change {cand_auc - base_auc:+.4f})")
    print(f"balance_trend coefficient:          {coefs[-1]:+.4f}  (largest input: {abs(coefs).max():.3f})")


if __name__ == "__main__":
    main()
