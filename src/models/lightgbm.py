"""LightGBM regression and two-sided ranking baselines for JPX."""

import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor

from ..baselines import make_regression_pipeline
from ..features import FEATURE_COLUMNS
from ..preprocessing import KEY, TARGET


_TREE_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 500,
    "max_bin": 63,
    "random_state": 7,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def lightgbm_scores(
    train: pd.DataFrame, evaluation: pd.DataFrame, *, feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    num_leaves: int = 15, robust: bool = False,
) -> pd.Series:
    """Fit on the supplied history and predict the subsequent evaluation rows."""
    regressor = LGBMRegressor(objective="regression", **{**_TREE_PARAMETERS, "num_leaves": num_leaves})
    model = make_regression_pipeline(regressor, robust=robust).set_output(transform="pandas")
    columns = list(feature_columns)
    model.fit(train[columns], train[TARGET])
    return pd.Series(model.predict(evaluation[columns]), index=evaluation.index, name="LightGBM")


def lightgbm_rank_scores(
    train: pd.DataFrame, evaluation: pd.DataFrame, *, portfolio_size: int = 200,
) -> pd.Series:
    """Combine buy and sell ranking scores after normalizing within each signal date."""
    columns = list(FEATURE_COLUMNS)
    models = fit_rankers(train, columns, portfolio_size=portfolio_size)
    return ranker_scores(models, evaluation, columns).rename("LightGBM xếp hạng")


def fit_rankers(train, columns, *, portfolio_size=200, params=None):
    ordered = train.sort_values(KEY)
    groups = ordered.groupby("Date", sort=False).size().to_numpy()
    models = []
    for direction in (1, -1):
        target = direction * ordered[TARGET]
        rank = target.groupby(ordered["Date"], sort=False).rank(method="average", pct=True)
        relevance = rank.gt(0.90).astype("int32") + rank.gt(0.95).astype("int32")
        ranker = LGBMRanker(
            objective="lambdarank", label_gain=[0, 1, 2],
            lambdarank_truncation_level=portfolio_size + 3, **{**_TREE_PARAMETERS, **(params or {})},
        )
        model = make_regression_pipeline(ranker).set_output(transform="pandas")
        model.fit(ordered[columns], relevance, lgbmranker__group=groups)
        models.append(model)
    return models


def ranker_scores(models, evaluation, columns):
    percentiles = []
    for model in models:
        prediction = pd.Series(model.predict(evaluation[columns]), index=evaluation.index)
        percentiles.append(prediction.groupby(evaluation["Date"], sort=False).rank(method="average", pct=True))

    buy, sell = percentiles
    return buy - sell
