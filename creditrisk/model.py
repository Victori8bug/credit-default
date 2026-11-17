"""Train the default-probability models.

Logistic regression is the main model: it outputs a probability of default (PD)
and its coefficients can be read and explained. kNN is a benchmark that makes
no linearity assumption, used to check the simple model is not leaving much
predictive power on the table.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import FEATURES, TARGET
from .features import ENGINEERED_FEATURES, engineer

SEED = 42
TEST_SIZE = 0.30


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Hold out 30% of accounts. Stratify so train and test share the same default rate."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


_ENGINEER = FunctionTransformer(engineer)


def build_logit():
    """Main model: engineered features -> standardise -> logistic regression."""
    # Standardising puts every feature on the same scale, so coefficient sizes
    # are comparable and the default L2 penalty treats features evenly.
    return make_pipeline(_ENGINEER, StandardScaler(), LogisticRegression(max_iter=1000))


def build_logit_raw():
    """Baseline: the same model on the 19 untouched columns, kept to show what feature work adds."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def build_knn(k: int = 50):
    """Benchmark: default rate among the k most similar accounts in the training data."""
    # kNN is distance-based, so scaling is essential: otherwise one wide-ranging
    # column would dominate the distance.
    return make_pipeline(_ENGINEER, StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def _raw_plus_engineered(X):
    return pd.concat([X[FEATURES], engineer(X)], axis=1)


def build_gbm():
    """Challenger: gradient-boosted trees on all 28 columns.

    Hundreds of small decision trees, each correcting the errors of the ones
    before. Usually more accurate than logistic regression but far harder to
    explain, which is exactly the trade-off the app puts a number on.
    """
    return make_pipeline(
        FunctionTransformer(_raw_plus_engineered),
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=SEED),
    )


CHAMPION = "Logistic regression"
CHALLENGER = "Gradient boosting (challenger)"

MODELS = {
    "Logistic regression": build_logit,
    "Logistic regression (raw columns)": build_logit_raw,
    "k-nearest neighbours (k=50)": build_knn,
    CHALLENGER: build_gbm,
}


def predict_pd(model, X):
    """Probability of default for each row (column 1 of predict_proba)."""
    return model.predict_proba(X)[:, 1]


def logit_coefficients(model) -> pd.DataFrame:
    """Coefficients per one-standard-deviation change in each feature, largest first."""
    coefs = model.named_steps["logisticregression"].coef_[0]
    names = ENGINEERED_FEATURES if len(coefs) == len(ENGINEERED_FEATURES) else FEATURES
    out = pd.DataFrame({"feature": names, "coefficient": coefs})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
