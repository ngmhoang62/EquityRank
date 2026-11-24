"""Generate scores for rule-based and linear JPX baselines."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from .features import FEATURE_COLUMNS


RANDOM_SEEDS = (7, 19, 37, 73, 101)
RIDGE_ALPHA = 10_000.0


def rule_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Higher scores imply higher predicted returns."""
    scores = pd.DataFrame({
        "Reversal 1D": -frame["return_1"],
        "Momentum 20D": frame["return_20"],
    })
    for seed in RANDOM_SEEDS:
        scores[f"Random {seed}"] = np.random.default_rng(seed).random(len(frame))
    return scores


def regression_preprocessing(*, robust: bool = False):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if not robust:
        return (imputer, StandardScaler())
    return (
        imputer,
        RobustScaler(quantile_range=(10, 90)),
        FunctionTransformer(np.clip, kw_args={"a_min": -8., "a_max": 8.}, feature_names_out="one-to-one"),
    )


def make_regression_pipeline(regressor, *, robust: bool = False):
    """Give each regressor the same train-fitted feature transformations."""
    return make_pipeline(*regression_preprocessing(robust=robust), regressor)


def fit_ridge(
    train: pd.DataFrame, *, alpha: float = RIDGE_ALPHA, target_percentiles: bool = False,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS, robust: bool = False,
):
    """Fit the complete Ridge pipeline on labeled training rows."""
    target = train["Target"]
    if target_percentiles:
        target = train.groupby("Date", sort=False)["Target"].rank(method="average", pct=True)
    model = make_regression_pipeline(Ridge(alpha=alpha, solver="cholesky"), robust=robust)
    return model.fit(train[list(feature_columns)], target)


def ridge_scores(
    train: pd.DataFrame, evaluation: pd.DataFrame, *, target_percentiles: bool = False,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS, alpha: float = RIDGE_ALPHA, robust: bool = False,
) -> pd.Series:
    """Fit every preprocessing step using only the corresponding training set."""
    model = fit_ridge(
        train, alpha=alpha, target_percentiles=target_percentiles, feature_columns=feature_columns, robust=robust,
    )
    predictions = model.predict(evaluation[list(feature_columns)])
    return pd.Series(predictions, index=evaluation.index, name="Ridge")
