"""Kiểm tra đặc trưng tương đối và tổ hợp tín hiệu trên validation v2."""

import argparse
from pathlib import Path
import sys

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics import calendar_blocks, paired_calendar_intervals
from src.evaluation import evaluate_ranked_books
from src.features import FEATURE_COLUMNS, SESSION_FEATURE_COLUMNS
from src.preprocessing import KEY, TARGET
from src.release_data import FIN_COLS
from src.release_experiment import (
    CUTOFF, FINANCE, INTRADAY, economic_scores, research_inputs, signal_window, train_before, write_csv,
)
from src.release_models import FEATURES, LGB
from src.splitting import development_windows


def augmentation_results(grid, fold_map, common, intervals, output):
    """Report paired gaps and window-by-window results for the augmentation study."""
    paired = grid.where(grid.notna().all(axis=1)).sub(grid[FINANCE], axis=0).mul(10_000)
    rng = np.random.default_rng(31)
    samples = np.concatenate([
        calendar_blocks(part.to_numpy(), 5, 2000, rng) for _, part in paired.groupby(fold_map)
    ], axis=1)
    spread_limits = np.nanquantile(np.nanmean(samples, axis=1), [.025, .975], axis=0)
    summary = pd.DataFrame({
        "Sessions": len(common), "Sharpe": common.mean() / common.std(),
        "Spread bps": common.mean() * 10_000,
        "Spread gap low": spread_limits[0], "Spread gap high": spread_limits[1],
    })
    summary = summary.join(intervals[["Low", "High"]]).rename(columns={"Low": "Sharpe gap low", "High": "Sharpe gap high"})
    summary = summary.rename_axis("Model").reset_index()
    by_fold = []
    for start, window in common.groupby(fold_map):
        by_fold.append(pd.DataFrame({
            "Fold": start.date(), "Sessions": len(window), "Sharpe": window.mean() / window.std(),
            "Spread bps": window.mean() * 10_000,
        }).rename_axis("Model").reset_index())
    write_csv(output / "validation_ohlcv_augmentation.csv", summary)
    write_csv(output / "validation_ohlcv_augmentation_folds.csv", pd.concat(by_fold, ignore_index=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--augment-ohlcv", action="store_true",
                        help="Giữ OHLCV và công bố, rồi thêm các nhóm đặc trưng tương đối.")
    args = parser.parse_args()
    output = args.output
    original = pd.read_csv(output / "development_daily.csv", parse_dates=["Date"])
    original = original.pivot(index="Date", columns="Model", values="Spread")
    prices = pd.read_pickle(output / "prices.pkl")
    disclosures = pd.read_pickle(output / "financial.pkl")
    panel, _, fit, labels, ends = research_inputs(prices, disclosures)
    del prices, disclosures

    base_inputs = list(FEATURES["finance"])
    relative = [*FEATURE_COLUMNS, *SESSION_FEATURE_COLUMNS]
    inputs = {
        FINANCE: base_inputs,
        "LightGBM relative": relative,
        "LightGBM relative + finance": [*relative, *FIN_COLS],
    }
    if args.augment_ohlcv:
        activity = ["return_1", "volatility_20", "relative_turnover_20"]
        inputs = {
            FINANCE: base_inputs,
            "LightGBM finance + session": [*base_inputs, *SESSION_FEATURE_COLUMNS],
            "LightGBM finance + activity": [*base_inputs, *activity],
            "LightGBM finance + session + activity": [*base_inputs, *SESSION_FEATURE_COLUMNS, *activity],
        }
    results, calendars = [], []
    for start, end in development_windows(str(CUTOFF.date())):
        train = train_before(fit, start)
        signals = signal_window(panel, ends, start, end)
        scores = economic_scores(signals)[[INTRADAY]].copy()
        for name, columns in inputs.items():
            model = LGBMRegressor(**LGB).fit(train[columns], train[TARGET])
            scores[name] = model.predict(signals[columns])
        if not args.augment_ohlcv:
            percentiles = scores[[FINANCE, INTRADAY]].groupby(signals["Date"]).rank(pct=True)
            scores["LightGBM finance + reversal 50/50"] = percentiles.mean(axis=1)
        _, daily = evaluate_ranked_books(signals[KEY], scores, labels)
        results.append(daily)
        dates = ends.index[(ends.index >= start) & ends.lt(end)]
        calendars.append(pd.Series(start, index=dates))
        print(f"{start.date()}: {daily['Date'].nunique()} validation sessions", flush=True)

    table = pd.concat(results, ignore_index=True)
    grid = table.pivot(index="Date", columns="Model", values="Spread")
    for name in (FINANCE, INTRADAY):
        np.testing.assert_allclose(grid[name], original[name].reindex(grid.index), rtol=1e-10, atol=1e-12)
    fold_map = pd.concat(calendars)
    grid = grid.reindex(fold_map.index)
    common = grid.dropna()
    intervals = paired_calendar_intervals(grid, fold_map, FINANCE).query("Block == 5").set_index("Model")
    if args.augment_ohlcv:
        summary = augmentation_results(grid, fold_map, common, intervals, output)
    else:
        summary = pd.DataFrame({"Sessions": len(common), "Sharpe": common.mean() / common.std(),
                                "Spread bps": common.mean() * 10_000})
        summary = summary.join(intervals[["Low", "High"]]).rename_axis("Model").reset_index()
        write_csv(output / "validation_signal_checks.csv", summary)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
