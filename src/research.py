"""Time-aware development experiments on preprocessing, inputs, and models."""

import pandas as pd

from .baselines import RIDGE_ALPHA, ridge_scores
from .evaluation import evaluate_scores
from .features import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, SESSION_FEATURE_COLUMNS
from .models.lightgbm import lightgbm_scores
from .preprocessing import LABEL_END


SESSION_INPUTS = (*FEATURE_COLUMNS, *SESSION_FEATURE_COLUMNS)
RESEARCH_INPUTS = (*SESSION_INPUTS, *MARKET_FEATURE_COLUMNS)
RIDGE_BASE = "Ridge · 6"


def inner_training_split(frame: pd.DataFrame, sessions: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the last complete signal sessions of the outer training history."""
    dates = pd.DatetimeIndex(frame["Date"].unique()).sort_values()
    if not 0 < sessions < len(dates):
        raise ValueError("Inner evaluation needs fewer sessions than the available history.")
    start = dates[-sessions]
    train = frame.loc[frame[LABEL_END].lt(start)]
    if train.empty:
        raise ValueError("No training rows remain after removing overlapping labels.")
    return train, frame.loc[frame["Date"].ge(start)]


def _search_parameters(fit: pd.DataFrame, *, portfolio_size: int):
    inner_fit, inner_eval = inner_training_split(fit)
    inner_eval = inner_eval.dropna(subset=["return_1", "return_20"])
    params = {"Ridge": (RIDGE_ALPHA, 100., 1_000_000.), "LightGBM": (15, 31)}
    predictions = {}
    for alpha in params["Ridge"]:
        predictions[f"Ridge {alpha:g}"] = ridge_scores(
            inner_fit, inner_eval, feature_columns=RESEARCH_INPUTS, robust=True, alpha=alpha,
        )
    for leaves in params["LightGBM"]:
        predictions[f"LightGBM {leaves:g}"] = lightgbm_scores(
            inner_fit, inner_eval, feature_columns=RESEARCH_INPUTS, robust=True, num_leaves=leaves,
        )
    metrics, _, _ = evaluate_scores(inner_eval, pd.DataFrame(predictions), portfolio_size=portfolio_size)
    spread = metrics["Spread (bps/phiên)"]
    alpha = max(params["Ridge"], key=lambda value: spread[f"Ridge {value:g}"])
    leaves = max(params["LightGBM"], key=lambda value: spread[f"LightGBM {value:g}"])
    return alpha, leaves, spread


def fit_research_tabular(fit: pd.DataFrame, evaluation: pd.DataFrame, *, portfolio_size: int):
    """Fit controlled tabular variants, selecting parameters inside the outer train."""
    if fit[LABEL_END].max() >= evaluation["Date"].min():
        raise ValueError("The outer training window contains unrealized labels.")
    alpha, leaves, inner_spread = _search_parameters(fit, portfolio_size=portfolio_size)
    ridge_wide = ridge_scores(fit, evaluation, robust=True, feature_columns=RESEARCH_INPUTS)
    tree_wide = lightgbm_scores(fit, evaluation, robust=True, feature_columns=RESEARCH_INPUTS)
    scores = pd.DataFrame({
        RIDGE_BASE: ridge_scores(fit, evaluation),
        "Ridge · 6 robust": ridge_scores(fit, evaluation, robust=True),
        "Ridge · 8 phiên": ridge_scores(fit, evaluation, robust=True, feature_columns=SESSION_INPUTS),
        "Ridge · 11": ridge_wide,
        "Ridge · 11 tinh chỉnh": ridge_wide if alpha == RIDGE_ALPHA else ridge_scores(
            fit, evaluation, robust=True, feature_columns=RESEARCH_INPUTS, alpha=alpha,
        ),
        "LightGBM · 6": lightgbm_scores(fit, evaluation),
        "LightGBM · 11": tree_wide,
        "LightGBM · 11 tinh chỉnh": tree_wide if leaves == 15 else lightgbm_scores(
            fit, evaluation, robust=True, feature_columns=RESEARCH_INPUTS, num_leaves=leaves,
        ),
    })
    return scores, {"alpha": alpha, "leaves": leaves}, inner_spread


def evaluate_research_scores(
    fit: pd.DataFrame, evaluation: pd.DataFrame, scores: pd.DataFrame, *, portfolio_size: int,
):
    """Score the same rows for every model and annotate regimes using a train threshold."""
    metrics, coverage, daily = evaluate_scores(evaluation, scores, portfolio_size=portfolio_size)
    expected = len(evaluation)
    if not coverage["Số mã-phiên có điểm"].eq(expected).all():
        raise ValueError("All models must score the same eligible observations.")

    fit_dispersion = fit.groupby("Date")["market_dispersion"].first()
    eval_dispersion = evaluation.groupby("Date")["market_dispersion"].first()
    high_dispersion = eval_dispersion.gt(fit_dispersion.median()).rename("High dispersion")
    daily = daily.join(high_dispersion, on="Date")
    return metrics, daily
