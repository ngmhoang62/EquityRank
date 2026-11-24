"""Model specifications shared by development and release inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from xgboost import XGBRegressor

from .baselines import make_regression_pipeline
from .features import FEATURE_COLUMNS
from .models.lightgbm import fit_rankers, ranker_scores
from .preprocessing import TARGET
from .release_data import FIN_EVENTS, LEGACY_FIN, PRICE_COLS, RATIOS, REPORT_AGES, REPORT_MISSING, TREND_COLS
from .research import RESEARCH_INPUTS


BASE = "Ridge six"
FINANCE_CONFIG = Path(__file__).resolve().parents[1] / "configs/lightgbm_finance.json"
finance_settings = json.loads(FINANCE_CONFIG.read_text(encoding="utf-8"))
FINANCE = finance_settings["model"]
TREES = {"lightgbm", "xgboost"}
NEURAL = {"mlp", "tcn", "gru", "tcn_shuffled"}
TREE = {"n_estimators": 100, "learning_rate": .05, "n_jobs": 4, "random_state": 7}
LGB = finance_settings["parameters"]
FEATURES = {
    "six": FEATURE_COLUMNS, "ohlcv": PRICE_COLS,
    "log": tuple("log_" + name for name in PRICE_COLS),
    "six_trend": (*FEATURE_COLUMNS, *TREND_COLS), "ohlcv_trend": (*PRICE_COLS, *TREND_COLS),
    "finance": tuple(finance_settings["features"]), "neural": RESEARCH_INPUTS,
    "finance_ratios": (*PRICE_COLS, *RATIOS),
    "finance_availability": (*PRICE_COLS, *REPORT_AGES, *REPORT_MISSING),
    "finance_events": (*PRICE_COLS, *FIN_EVENTS),
    "finance_legacy": (*PRICE_COLS, *LEGACY_FIN),
    "coverage": (*FEATURE_COLUMNS, "return_coverage_20", "volatility_partial_20"),
    "percentiles": tuple("percentile_" + name for name in FEATURE_COLUMNS),
}
SPECS = {
    BASE: ("ridge", "six", [{"alpha": 10000}]),
    "LightGBM OHLCV": ("lightgbm", "ohlcv", [{}]),
    "OLS log OHLCV": ("ols", "log", [{}]),
    "OLS OHLCV": ("ols", "ohlcv", [{}]),
    "Ridge tuned": ("ridge", "six", [{"alpha": a} for a in (10000, 100, 1000000)]),
    "Elastic Net": ("elastic", "six", [{"alpha": a} for a in (1e-6, 1e-5, 1e-4)]),
    "Ridge centered": ("ridge_centered", "six", [{"alpha": 10000}]),
    "Ridge feature percentiles": ("ridge", "percentiles", [{"alpha": 10000}]),
    "Ridge target percentiles": ("ridge_percentile", "six", [{"alpha": 10000}]),
    "Ridge coverage": ("ridge", "coverage", [{"alpha": 10000}]),
    "Ridge rolling 504": ("ridge", "six", [{"alpha": 10000}]),
    "Ridge without outage labels": ("ridge", "six", [{"alpha": 10000}]),
    "Ridge six + trend": ("ridge", "six_trend", [{"alpha": 10000}]),
    "Ridge OHLCV + trend": ("ridge", "ohlcv_trend", [{"alpha": 10000}]),
    "LightGBM six + trend": ("lightgbm", "six_trend", [{}]),
    "LightGBM OHLCV + trend": ("lightgbm", "ohlcv_trend", [{}]),
    "LightGBM tuned": ("lightgbm", "ohlcv", [
        {"num_leaves": leaves, "min_child_samples": samples} for leaves in (15, 31) for samples in (500, 1000)
    ]),
    "XGBoost tuned": ("xgboost", "ohlcv", [
        {"max_depth": depth, "min_child_weight": weight} for depth in (3, 5) for weight in (50, 200)
    ]),
    FINANCE: ("lightgbm", "finance", [{}]),
    "LightGBM ratios": ("lightgbm", "finance_ratios", [{}]),
    "LightGBM report availability": ("lightgbm", "finance_availability", [{}]),
    "LightGBM report events": ("lightgbm", "finance_events", [{}]),
    "LightGBM legacy finance": ("lightgbm", "finance_legacy", [{}]),
    "LightGBM finance median": ("lightgbm", "finance", [{"impute": True}]),
    "LightGBM early stop": ("lightgbm", "ohlcv", [{}, {"early_stop": True, "n_estimators": 300}]),
    "XGBoost early stop": ("xgboost", "ohlcv", [{"max_depth": 3, "min_child_weight": 50},
        {"max_depth": 3, "min_child_weight": 50, "early_stop": True, "n_estimators": 300}]),
    "Huber": ("lightgbm", "six", [{"objective": "huber", "alpha": .9}]),
    "Median regression": ("lightgbm", "six", [{"objective": "quantile", "alpha": .5}]),
    "LambdaRank two sides": ("ranker", "six", [{}]),
    "Ridge short guard": ("guard", "six", [{"penalty": p} for p in (0., .25, .5, 1.)]),
    **{name: (name.lower(), "neural", [{}]) for name in ("MLP", "TCN", "GRU")},
    "TCN shuffled": ("tcn_shuffled", "neural", [{}]),
}
AUXILIARY_MODELS = {
    "LightGBM ratios", "LightGBM report availability", "LightGBM report events",
    "LightGBM legacy finance", "LightGBM finance median", "Ridge rolling 504", "TCN shuffled",
    "Ridge without outage labels",
}
CANDIDATES = tuple(name for name in SPECS if name not in AUXILIARY_MODELS)


def guard_scores(keys, ridge, risk, *, penalty, size):
    if penalty == 0.:
        return ridge
    result = ridge.copy().astype(float)
    frame = keys.assign(score=ridge, risk=risk)
    for _, group in frame.groupby("Date", sort=False):
        original = group.sort_values(["score", "SecuritiesCode"], ascending=[False, True])
        pool = original.iloc[size:].tail(2 * size)
        if len(pool) < size:
            continue
        priority = pool["score"].rank(pct=True) + penalty * pool["risk"].rank(pct=True)
        short = pool.assign(priority=priority).sort_values(
            ["priority", "risk", "SecuritiesCode"], ascending=[True, True, False],
        ).index[:size]
        remaining = original.index[~original.index.isin(short)]
        order = remaining.append(short[::-1])
        result.loc[order] = -np.arange(len(order), dtype=float)
    return result


def fit_candidate(name, fit, params, history, device, *, validation=None):
    family, feature_set, _ = SPECS[name]
    columns = list(FEATURES[feature_set])
    if name == "Ridge rolling 504":
        recent = fit["Date"].drop_duplicates().sort_values().iloc[-504:]
        fit = fit.loc[fit["Date"].ge(recent.iloc[0])]
    if family in NEURAL:
        from .release_neural import fit_neural

        return fit_neural(family, fit, columns, history, device)
    target = fit[TARGET]
    if family == "ridge_centered":
        target = target - target.groupby(fit["Date"]).transform("mean")
    if family == "ridge_percentile":
        target = target.groupby(fit["Date"]).rank(pct=True)
    if family == "ranker":
        return {"family": family, "columns": columns, "rankers": fit_rankers(fit, columns, params=params)}
    params = dict(params)
    median = params.pop("impute", False)
    stopping = params.pop("early_stop", False)
    fit_args = {}
    if family.startswith("ridge") or family == "guard":
        estimator = Ridge(alpha=params.get("alpha", 10000), solver="cholesky")
    elif family == "ols":
        estimator = LinearRegression()
    elif family == "elastic":
        estimator = ElasticNet(l1_ratio=.5, max_iter=1000, **params)
    elif family == "xgboost":
        estimator = XGBRegressor(**{**TREE, **params}, tree_method="hist", max_bin=63, verbosity=0)
        if stopping:
            estimator.set_params(early_stopping_rounds=20)
            fit_args["verbose"] = False
    else:
        estimator = LGBMRegressor(**{**LGB, **params})
    if stopping:
        if validation is None or validation.empty:
            raise ValueError("Early stopping requires completed inner validation labels.")
        fit_args["eval_set"] = [(validation[columns], validation[TARGET])]
        if family == "lightgbm":
            fit_args["callbacks"] = [early_stopping(20, verbose=False)]
    predictor = estimator if family in TREES and not median else make_regression_pipeline(estimator)
    if family == "lightgbm" and median:
        predictor.set_output(transform="pandas")
    predictor.fit(fit[columns], target, **fit_args)
    result = {"columns": columns, "predictor": predictor, "family": family}
    if stopping:
        result["best_iteration"] = (
            estimator.best_iteration_ if family == "lightgbm" else estimator.best_iteration + 1
        )
    if family == "guard":
        relevance = target.groupby(fit["Date"]).rank(pct=True).gt(.9).astype(int)
        classifier = make_regression_pipeline(LGBMClassifier(**LGB)).set_output(transform="pandas")
        result["guard"] = classifier.fit(fit[columns], relevance)
    return result


def predict_candidate(bundle, signals, history, device, params, size, *, seed_scores=False):
    if bundle["family"] in NEURAL:
        from .release_neural import predict_neural

        seeds = predict_neural(bundle, signals, history, device)
        mean = seeds.mean(axis=1)
        return (mean, seeds) if seed_scores else mean
    if bundle["family"] == "ranker":
        return ranker_scores(bundle["rankers"], signals, bundle["columns"])
    values = signals[bundle["columns"]]
    score = pd.Series(bundle["predictor"].predict(values), index=signals.index)
    if bundle["family"] == "guard":
        risk = pd.Series(bundle["guard"].predict_proba(values)[:, 1], index=signals.index)
        score = guard_scores(signals, score, risk, penalty=params["penalty"], size=size)
    return score
