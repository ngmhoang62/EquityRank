"""Train, serialize, rank, and evaluate the fixed JPX Ridge configuration."""

from collections.abc import Callable
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from .baselines import fit_ridge
from .evaluation import jpx_sharpe
from .features import FEATURE_COLUMNS, build_price_features
from .liquidity import cost_sensitivity, simulate_ranked_portfolio
from .preprocessing import KEY, LABEL_END, OHLC, TARGET, load_stock_prices, prepare_supervised


PRICE_INPUTS = [*KEY, *OHLC, "Volume", "AdjustmentFactor"]
FORMAT_VERSION = 1


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["features"]) != FEATURE_COLUMNS:
        raise ValueError("Configuration features must match the Ridge feature order.")
    if not set(config["required_signal_features"]).issubset(FEATURE_COLUMNS):
        raise ValueError("Unknown required signal feature.")
    pd.Timestamp(config["test_start"])
    return config


def fit_final_model(prices: pd.DataFrame, config: dict) -> dict:
    """Fit all steps strictly before the frozen test start, including label realization."""
    cutoff = pd.Timestamp(config["test_start"])
    history = prices.loc[prices["Date"].lt(cutoff)]
    prepared, _ = prepare_supervised(history)
    if prepared.empty:
        raise ValueError("No completed labels before the training cutoff.")
    computed = build_price_features(history[PRICE_INPUTS])
    train = prepared[[*KEY, TARGET]].merge(computed, on=KEY, validate="one_to_one")
    model = fit_ridge(train, alpha=config["alpha"])
    metadata = {
        "training_rows": len(train),
        "training_sessions": int(train["Date"].nunique()),
        "first_signal": train["Date"].min().date().isoformat(),
        "last_signal": train["Date"].max().date().isoformat(),
        "last_label_end": prepared[LABEL_END].max().date().isoformat(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
    }
    return {"format_version": FORMAT_VERSION, "config": config, "metadata": metadata, "model": model}


def train_from_file(path: Path, config: dict) -> dict:
    bundle = fit_final_model(load_stock_prices(path), config)
    with path.open("rb") as data:
        bundle["metadata"]["input_sha256"] = file_digest(data, "sha256").hexdigest()
    return bundle


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        write(temporary)
        temporary.replace(path)


def save_model(bundle: dict, path: Path) -> None:
    atomic_write(path, lambda temporary: joblib.dump(bundle, temporary))


def load_model(path: Path) -> dict:
    """Load an artifact created by this project from a trusted local path."""
    bundle = joblib.load(path)
    if bundle["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format.")
    return bundle


def rank_prices(bundle: dict, prices: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """Rank using OHLCV and calendar history. Rank zero is the highest prediction."""
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end else prices["Date"].max()
    if start_date < pd.Timestamp(bundle["config"]["test_start"]) or end_date < start_date:
        raise ValueError("Ranking dates must follow model fit and form a valid interval.")
    history = prices.loc[prices["Date"].le(end_date), PRICE_INPUTS]
    features = build_price_features(history)
    valid_bar = history[OHLC].gt(0).all(axis=1) & history["Volume"].gt(0)
    candidates = history.loc[valid_bar & history["Date"].ge(start_date), KEY]
    inputs = candidates.merge(features, on=KEY, validate="one_to_one").dropna(
        subset=bundle["config"]["required_signal_features"],
    )
    if inputs.empty:
        raise ValueError("No eligible securities in the requested interval. Check input history.")
    predictions = inputs[KEY].copy()
    predictions["Score"] = bundle["model"].predict(inputs[bundle["config"]["features"]])
    predictions = predictions.sort_values(
        ["Date", "Score", "SecuritiesCode"], ascending=[True, False, True], ignore_index=True,
    )
    predictions["Rank"] = predictions.groupby("Date").cumcount()
    return predictions


def evaluation_calendar(prices: pd.DataFrame, config: dict) -> pd.DatetimeIndex:
    """Use complete signal horizons from the observed calendar before evaluating returns."""
    dates = pd.DatetimeIndex(prices["Date"].unique(), name="Date").sort_values()[:-2]
    return dates[dates >= config["test_start"]]


def evaluate_predictions(
    predictions: pd.DataFrame, labels: pd.DataFrame, sessions: pd.DatetimeIndex, config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join labels after ranking, price the original positions, and score random controls."""
    ranked = predictions.loc[predictions["Date"].isin(sessions)].sort_values(KEY).reset_index(drop=True)
    ranks = ranked.sort_values(["Date", "Rank"])
    if not ranks["Rank"].eq(ranks.groupby("Date").cumcount()).all():
        raise ValueError("Expected consecutive, unique ranks starting at zero in every session.")
    observed = ranked.merge(labels[[*KEY, TARGET]], on=KEY, how="left", validate="one_to_one")
    known = np.isfinite(observed[TARGET])
    daily_tables, rows, cost_tables = {}, {}, {}
    model_ranks = {"Ridge": observed["Rank"]}
    for seed in config["random_seeds"]:
        random = pd.Series(np.random.default_rng(seed).random(len(observed)), index=observed.index)
        model_ranks[f"Random {seed}"] = random.groupby(observed["Date"]).rank(
            ascending=False, method="first",
        ).sub(1)
    for name, model_rank in model_ranks.items():
        daily = simulate_ranked_portfolio(
            observed.assign(Rank=model_rank), sessions, config["portfolio_size"],
        )
        ic_frame = observed.loc[known, ["Date", TARGET]].assign(Score=-model_rank.loc[known])
        per_date_ic = ic_frame.groupby("Date")[["Score", TARGET]].apply(
            lambda group: group["Score"].rank().corr(group[TARGET].rank())
            if group["Score"].nunique() > 1 and group[TARGET].nunique() > 1 else np.nan
        )
        daily["IC"] = per_date_ic
        trades = daily["Leg size"].gt(0)
        valid_gross = daily.loc[trades, "Gross bps"]
        rows[name] = {
            "Predictions": len(observed),
            "Known labels": int(known.sum()),
            "Sessions": len(sessions),
            "Trading sessions": int(trades.sum()),
            "Priced trading sessions": int(valid_gross.notna().sum()),
            "Missing selected labels": int(daily["Missing selected labels"].sum()),
            "IC sessions": int(daily["IC"].count()),
            "Mean IC": daily["IC"].mean(),
            "Gross bps": daily["Gross bps"].mean(skipna=False),
            "JPX-style Sharpe": jpx_sharpe(valid_gross),
            "Mean turnover": daily["Traded notional"].mean(),
        }
        daily_tables[name] = daily
        cost_tables[name] = cost_sensitivity(daily, config["cost_rates_bps"])
    daily = pd.concat(daily_tables, names=["Model"]).reset_index()
    summary = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model")
    costs = pd.concat(cost_tables, names=["Model"])
    return daily, summary, costs


def paired_portfolio_summary(daily: pd.DataFrame, rates_bps) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare priced portfolios on shared sessions using the original turnover."""
    active = daily.loc[daily["Leg size"].gt(0)]
    gross = active.pivot(index="Date", columns="Model", values="Gross bps")
    gross = gross.reindex(columns=daily["Model"].unique())
    shared = gross.index[np.isfinite(gross).all(axis=1)]
    matched = daily.loc[daily["Date"].isin(shared)].copy()
    rows, cost_tables = {}, {}
    for model, group in daily.groupby("Model", sort=False):
        group = group.loc[group["Date"].isin(shared)]
        mean = group["Gross bps"].mean()
        rows[model] = {
            "Priced sessions": len(group),
            "Mean IC": group["IC"].mean(),
            "Gross bps": mean,
            "JPX-style Sharpe": jpx_sharpe(group["Gross bps"]),
            "Mean turnover": group["Traded notional"].mean(),
        }
        cost_tables[model] = cost_sensitivity(group, rates_bps)
    summary = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model")
    return matched, summary, pd.concat(cost_tables, names=["Model"])
