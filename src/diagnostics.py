"""Inspect a ranking strategy alongside conditional random rankings."""

import numpy as np
import pandas as pd

from .evaluation import common_scoring_rows, rank_weights


def calendar_blocks(values, block_size, resamples, rng):
    """Sample contiguous calendar blocks, leaving missing dates in place."""
    width = min(block_size, len(values))
    if width < 1:
        raise ValueError("A bootstrap block needs at least one calendar position.")
    count = (len(values) + width - 1) // width
    starts = rng.integers(len(values) - width + 1, size=(resamples, count))
    indices = (starts[..., None] + np.arange(width)).reshape(resamples, -1)[:, :len(values)]
    return values[indices]


def paired_calendar_intervals(spread, fold_map, baseline, *, resamples=2000):
    """Estimate pointwise and simultaneous paired Sharpe gaps from common spreads."""
    spread = spread.sort_index()
    spread = spread.where(spread.notna().all(axis=1), np.nan)
    observed = spread.mean().div(spread.std()).to_numpy()
    baseline_idx = spread.columns.get_loc(baseline)
    observed -= observed[baseline_idx]
    rows = []
    for width in (5, 10, 20):
        rng = np.random.default_rng(31)
        samples = np.concatenate([
            calendar_blocks(part.to_numpy(), width, resamples, rng)
            for _, part in spread.groupby(fold_map)
        ], axis=1)
        count = np.isfinite(samples).sum(axis=1)
        total = np.nansum(samples, axis=1)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        squared = np.nansum((samples - mean[:, None, :]) ** 2, axis=1)
        variance = np.divide(squared, count - 1, out=np.full_like(squared, np.nan), where=count > 1)
        sharpe = np.divide(mean, np.sqrt(variance), out=np.full_like(mean, np.nan), where=variance > 0)
        deltas = sharpe - sharpe[:, [baseline_idx]]
        finite = np.isfinite(deltas).all(axis=1)
        draws = deltas[finite]
        if not len(draws):
            low = high = np.full(len(spread.columns), np.nan)
            band = np.nan
        else:
            low, high = np.quantile(draws, [.025, .975], axis=0)
            band = np.quantile(np.abs(draws - observed).max(axis=1), .95)
        rows.extend(zip(spread.columns, [width] * len(low), low, high, observed - band, observed + band,
                        [len(draws)] * len(low), [int(spread.notna().all(axis=1).sum())] * len(low)))
    return pd.DataFrame(rows, columns=["Model", "Block", "Low", "High", "Simultaneous low", "Simultaneous high",
                                      "Bootstrap draws", "Common score sessions"])


def paired_sharpe_interval(
    paired: pd.DataFrame, block_size: int = 5, n_resamples: int = 2_000, seed: int = 31,
) -> tuple[float, float]:
    """Resample the same dates for two models and return the Sharpe-difference interval."""
    values = paired.sort_index().to_numpy(dtype=float)
    if values.shape[1] != 2 or not 1 <= block_size <= len(values) or n_resamples < 1:
        raise ValueError("Expected two models and a block fitting the paired calendar.")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Use complete paired daily spreads.")
    rng = np.random.default_rng(seed)
    count = (len(values) + block_size - 1) // block_size
    starts = rng.integers(len(values) - block_size + 1, size=(n_resamples, count))
    indices = (starts[..., None] + np.arange(block_size)).reshape(n_resamples, -1)[:, :len(values)]
    samples = values[indices]
    std = samples.std(axis=1, ddof=1)
    sharpe = np.divide(samples.mean(axis=1), std, out=np.full_like(std, np.nan), where=std > 0)
    delta = sharpe[:, 0] - sharpe[:, 1]
    defined = delta[np.isfinite(delta)]
    return tuple(np.quantile(defined, [0.025, 0.975])) if len(defined) else (np.nan, np.nan)


def short_tail_diagnostics(
    frame: pd.DataFrame, scores: pd.DataFrame, *, portfolio_size: int = 200, tail_size: int = 5,
) -> pd.DataFrame:
    """Decompose the same weighted short portfolio used in the spread metric."""
    if not 1 <= tail_size <= portfolio_size:
        raise ValueError("The tail size must fit within the short portfolio.")
    weights = rank_weights(portfolio_size)
    samples, _ = common_scoring_rows(frame, scores)
    rows = []
    for date, session in samples.groupby("Date", sort=True):
        if len(session) < 2 * portfolio_size:
            continue
        target = session["Target"].to_numpy()
        relative = target - target.mean()
        for name in scores:
            order = np.argsort(-session[name].to_numpy(), kind="stable")[-portfolio_size:][::-1]
            selected = relative[order]
            weighted = selected * weights
            losses = np.maximum(weighted, 0)
            largest_losses = np.partition(losses, -tail_size)[-tail_size:].sum()
            rows.append((date, name, -weighted.sum(), largest_losses, (selected > 0).mean(), np.median(selected)))
    return pd.DataFrame(rows, columns=[
        "Date", "Baseline", "Short excess", "Largest loss excess", "Share above market", "Median excess",
    ])


def rank_diagnostics(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    model: str,
    portfolio_size: int = 200,
    n_groups: int = 10,
    n_random: int = 200,
    seed: int = 2021,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return daily weighted legs, equal-weight rank groups and random-run metrics."""
    if n_groups < 2 or n_random < 1:
        raise ValueError("At least two rank groups and one random run are required.")
    weights = rank_weights(portfolio_size)
    samples, _ = common_scoring_rows(frame, scores)
    rng = np.random.default_rng(seed)
    daily, rank_groups, random_spreads = [], [], []
    for date, session in samples.groupby("Date", sort=True):
        if len(session) < max(2 * portfolio_size, n_groups):
            continue
        target = session["Target"].to_numpy()
        order = np.argsort(-session[model].to_numpy(), kind="stable")
        ranked = target[order]
        market = target.mean()
        top = weights @ ranked[:portfolio_size]
        bottom = weights @ ranked[-portfolio_size:][::-1]
        daily.append((date, len(session), market, top, bottom, top - bottom))

        for group, values in enumerate(np.array_split(ranked, n_groups), start=1):
            rank_groups.append((date, group, len(values), values.mean() - market))

        # Each row is a complete permutation, so the two random legs are disjoint.
        shuffled = rng.permuted(np.broadcast_to(target, (n_random, len(target))), axis=1)
        random_spreads.append(
            shuffled[:, :portfolio_size] @ weights - shuffled[:, -portfolio_size:][:, ::-1] @ weights
        )

    if not daily:
        raise ValueError("No common session is large enough for the requested groups.")
    daily_frame = pd.DataFrame(
        daily, columns=["Date", "Securities", "Market", "Top", "Bottom", "Spread"],
    ).set_index("Date")
    group_frame = pd.DataFrame(
        rank_groups, columns=["Date", "Rank group", "Securities", "Excess"],
    ).set_index(["Date", "Rank group"])
    simulated = pd.DataFrame(np.stack(random_spreads))
    mean = simulated.mean()
    std = simulated.std()
    simulations = pd.DataFrame({"Spread bps": 10_000 * mean, "Sharpe": mean / std.where(std.gt(0))})
    return daily_frame, group_frame, simulations


def spread_mean_interval(
    differences: pd.Series, block_size: int = 5, n_resamples: int = 2_000, seed: int = 31,
) -> tuple[float, float]:
    """Estimate a percentile interval in bps using contiguous blocks of paired sessions."""
    values = differences.sort_index().to_numpy(dtype=float)
    if not 1 <= block_size <= len(values) or n_resamples < 1 or not np.isfinite(values).all():
        raise ValueError("Use finite paired differences and blocks fitting the available sessions.")
    block_count = (len(values) + block_size - 1) // block_size
    rng = np.random.default_rng(seed)
    starts = rng.integers(len(values) - block_size + 1, size=(n_resamples, block_count))
    indices = (starts[:, :, None] + np.arange(block_size)).reshape(n_resamples, -1)
    means = values[indices[:, :len(values)]].mean(axis=1)
    lower, upper = np.quantile(10_000 * means, [0.025, 0.975])
    return float(lower), float(upper)
