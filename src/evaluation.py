"""Evaluate ranking baselines on a shared set of JPX securities and sessions."""

import numpy as np
import pandas as pd


def common_scoring_rows(
    frame: pd.DataFrame, scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect finite scores and labels, then order rows deterministically."""
    if not scores.index.equals(frame.index) or not frame.index.is_unique:
        raise ValueError("Scores must align one-to-one with the evaluation rows.")
    known_target = np.isfinite(frame["Target"])
    available = scores.apply(np.isfinite).where(known_target, False)
    common = available.all(axis=1)
    keys = ["Date", "SecuritiesCode", "Target"]
    samples = pd.concat([frame.loc[common, keys], scores.loc[common]], axis=1)
    return samples.sort_values(["Date", "SecuritiesCode"]), available


def rank_weights(portfolio_size: int, ratio: float = 2.) -> np.ndarray:
    """Return normalized linear weights starting at the most extreme rank."""
    if portfolio_size < 1:
        raise ValueError("Portfolio size must be positive.")
    weights = np.linspace(ratio, 1.0, portfolio_size)
    return weights / weights.sum()


def jpx_sharpe(spread: pd.Series) -> float:
    """Unannualized mean divided by sample standard deviation."""
    if len(spread) < 2 or not np.isfinite(spread).all() or spread.nunique() < 2:
        return np.nan
    return float(spread.mean() / spread.std(ddof=1))


def weighted_legs(target_in_rank_order, portfolio_size: int, ratio: float = 2.) -> tuple[float, float]:
    """Score fixed positions. Missing outcomes propagate only in the selected leg."""
    target = np.asarray(target_in_rank_order, dtype=float)
    weights = rank_weights(portfolio_size, ratio)
    if len(target) < 2 * portfolio_size:
        return np.nan, np.nan
    target = np.where(np.isfinite(target), target, np.nan)
    return weights @ target[:portfolio_size], weights @ target[-portfolio_size:][::-1]


def rank_scores(signals: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """Rank a fixed universe using only signal-time keys and finite scores."""
    keys = ["Date", "SecuritiesCode"]
    if not score.index.equals(signals.index) or not signals.index.is_unique:
        raise ValueError("Scores must align one-to-one with the signal rows.")
    if signals[keys].isna().any().any() or signals.duplicated(keys).any():
        raise ValueError("Expected a unique, nonmissing date and security key.")
    if not np.isfinite(score.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("Every candidate must score the entire eligible universe.")
    ranked = signals[keys].assign(Score=score).sort_values(
        ["Date", "Score", "SecuritiesCode"], ascending=[True, False, True], ignore_index=True,
    )
    ranked["Rank"] = ranked.groupby("Date", sort=False).cumcount()
    return ranked


def evaluate_ranked_books(
    signals: pd.DataFrame, scores: pd.DataFrame, labels: pd.DataFrame, *, portfolio_size: int = 200,
    diagnostics: bool = False, weight_ratio: float = 2.,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fix the eligible universe and ranks before attaching realized outcomes."""
    keys = ["Date", "SecuritiesCode"]
    rank_weights(portfolio_size)
    if scores.empty or not scores.columns.is_unique:
        raise ValueError("Every candidate must score the entire eligible universe.")
    ranks, daily = [], []
    for model, score in scores.items():
        ranked = rank_scores(signals, score).assign(Model=model)
        ranks.append(ranked)
        priced = ranked.merge(labels[[*keys, "Target"]], on=keys, how="left", validate="one_to_one")
        for date, day in priced.groupby("Date", sort=False):
            target = day["Target"].to_numpy(dtype=float)
            known = np.isfinite(target)
            top, bottom = weighted_legs(target, portfolio_size, weight_ratio)
            active = len(day) >= 2 * portfolio_size
            selected = np.r_[target[:portfolio_size], target[-portfolio_size:]] if active else np.array([])
            paired = day.loc[known, ["Score", "Target"]]
            ic = (
                paired["Score"].rank().corr(paired["Target"].rank())
                if paired["Score"].nunique() > 1 and paired["Target"].nunique() > 1 else np.nan
            )
            row = [date, model, len(day), int(known.sum()), active,
                   int((~np.isfinite(selected)).sum()), top - bottom, top, bottom, ic]
            if diagnostics:
                market = target[known].mean() if known.any() else np.nan
                tail = np.maximum((target[-portfolio_size:][::-1] - market) * rank_weights(portfolio_size, weight_ratio), 0) if active else []
                tail_loss = np.sort(tail)[-5:].sum() if active and np.isfinite(tail).all() else np.nan
                groups = np.array_split(target, 10)
                row.extend([market, top - market, market - bottom, tail_loss])
                row.extend(g[np.isfinite(g)].mean() - market if np.isfinite(g).any() else np.nan for g in groups)
            daily.append(row)
    columns = [
        "Date", "Model", "Securities", "Known labels", "Active",
        "Missing selected labels", "Spread", "Top", "Bottom", "IC",
    ]
    if diagnostics:
        columns.extend(["Market", "Long excess", "Short excess", "Short top five loss",
                        *(f"Decile {i}" for i in range(1, 11))])
    return pd.concat(ranks, ignore_index=True), pd.DataFrame(daily, columns=columns)


def evaluate_scores(
    frame: pd.DataFrame, scores: pd.DataFrame, portfolio_size: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summary metrics, native coverage and daily results on common support."""
    weights = rank_weights(portfolio_size)
    samples, available = common_scoring_rows(frame, scores)
    counts_by_session = available.groupby(frame["Date"]).sum()
    coverage = pd.DataFrame({
        "Số mã-phiên có điểm": available.sum(),
        "Độ phủ (%)": 100 * available.sum() / np.isfinite(frame["Target"]).sum(),
        "Phiên đủ hai nhóm": counts_by_session.ge(2 * portfolio_size).sum(),
    })
    results = []
    for date, group in samples.groupby("Date", sort=True):
        if len(group) < 2 * portfolio_size:
            continue
        target = group["Target"]
        target_rank = target.rank()
        target_values = target.to_numpy()
        market_return = target_values.mean()
        distinct_targets = target_rank.nunique()
        for name in scores:
            signal = group[name]
            order = np.argsort(-signal.to_numpy(), kind="stable")
            long_return = weights @ target_values[order[:portfolio_size]]
            short_return = weights @ target_values[order[-portfolio_size:][::-1]]
            spread = 0.0 if distinct_targets == 1 else long_return - short_return
            ic = np.nan
            if distinct_targets > 1 and signal.nunique() > 1:
                ic = target_rank.corr(signal.rank())
            results.append((date, name, len(group), ic, spread, long_return, short_return, market_return))

    daily = pd.DataFrame(
        results, columns=["Date", "Baseline", "Số mã", "IC", "Spread", "Top", "Bottom", "Market"],
    )
    if daily.empty:
        raise ValueError("No common session contains enough scored securities.")
    grouped = daily.groupby("Baseline", sort=False)
    summary = grouped.agg(**{
        "Phiên spread": ("Spread", "count"),
        "Phiên IC hợp lệ": ("IC", "count"),
        "Số mã TB/phiên": ("Số mã", "mean"),
        "IC trung bình": ("IC", "mean"),
        "Spread (bps/phiên)": ("Spread", lambda values: 10_000 * values.mean()),
    })
    spread_std = grouped["Spread"].std()
    summary["Sharpe kiểu JPX"] = grouped["Spread"].mean().div(spread_std.where(spread_std.gt(0)))
    return summary, coverage, daily
