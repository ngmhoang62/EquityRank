"""Group securities by past liquidity and estimate turnover-based trading costs."""

import numpy as np
import pandas as pd

from .evaluation import rank_weights, weighted_legs


LIQUIDITY_GROUPS = ("Thấp", "Vừa", "Cao")
FULL_BOOK = "Toàn tập"
MID_BOOK = LIQUIDITY_GROUPS[1]


def liquidity_terciles(frame: pd.DataFrame) -> pd.Series:
    """Split each session's finite, positive liquidity estimates into three groups."""
    valid = frame["Liquidity"].gt(0) & np.isfinite(frame["Liquidity"])
    ordered = frame.loc[valid].sort_values(["Date", "Liquidity", "SecuritiesCode"])
    result = pd.Series(pd.NA, index=frame.index, dtype="string", name="Nhóm thanh khoản")
    for _, session in ordered.groupby("Date"):
        for group, indices in zip(LIQUIDITY_GROUPS, np.array_split(session.index.to_numpy(), 3)):
            result.loc[indices] = group
    return result


def portfolio_turnover(weights: pd.DataFrame) -> pd.Series:
    """Measure trades between target weights, including opening and final liquidation."""
    traded = weights.diff().abs().sum(axis=1)
    traded.iloc[0] = weights.iloc[0].abs().sum()
    traded.iloc[-1] += weights.iloc[-1].abs().sum()
    return traded.rename("Traded notional")


def liquidity_portfolios(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Build portfolios from signal-time scores, then look up realized labels."""
    available = _bucketed_scores(frame, sessions)
    books = [(FULL_BOOK, available)]
    books.extend((name, available.loc[available["Bucket"].eq(name)]) for name in LIQUIDITY_GROUPS)
    return _simulate_books(books, sessions)


def matched_liquidity_portfolios(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Compare rank selection with and without the middle tercile at identical leg sizes."""
    available = _bucketed_scores(frame, sessions).dropna(subset=["Bucket"])
    middle = available.loc[available["Bucket"].eq(MID_BOOK)]
    leg_sizes = middle.groupby("Date").size().floordiv(10)
    return _simulate_books([(FULL_BOOK, available), (MID_BOOK, middle)], sessions, leg_sizes)


def simulate_ranked_portfolio(
    frame: pd.DataFrame, sessions: pd.DatetimeIndex, portfolio_size: int,
) -> pd.DataFrame:
    """Evaluate a fixed-size portfolio with scores already produced at signal time."""
    scores = frame.assign(Score=-frame["Rank"])
    sizes = pd.Series(portfolio_size, index=sessions)
    return _simulate_books([(FULL_BOOK, scores)], sessions, sizes).xs(FULL_BOOK)


def _bucketed_scores(frame, sessions):
    available = frame.loc[np.isfinite(frame["Score"]) & frame["Date"].isin(sessions)].copy()
    available["Bucket"] = liquidity_terciles(available)
    return available


def _simulate_books(books, sessions, matched_sizes=None):
    results = []
    for name, rows in books:
        positions = []
        stats = []
        ranked = rows.sort_values(["Date", "Score", "SecuritiesCode"], ascending=[True, False, True])
        for date, day in ranked.groupby("Date"):
            if matched_sizes is not None:
                count = matched_sizes.get(date, 0)
            else:
                count = 200 if name == FULL_BOOK else len(day) // 10
            if count == 0 or len(day) < 2 * count:
                continue
            selected = pd.concat([day.iloc[:count], day.iloc[-count:][::-1]])
            leg = rank_weights(count)
            weights = np.concatenate([leg, -leg])
            target = selected["Target"].to_numpy()
            missing = int((~np.isfinite(target)).sum())
            top, bottom = weighted_legs(day["Target"], count)
            top, bottom = 10_000 * top, 10_000 * bottom
            gross = top - bottom
            market = 10_000 * day.loc[np.isfinite(day["Target"]), "Target"].mean()
            positions.append(selected[["Date", "SecuritiesCode"]].assign(Weight=weights))
            liquidity = day["Liquidity"].median() if "Liquidity" in day else np.nan
            stats.append((date, len(day), count, liquidity, missing, gross, top, bottom, market))

        if not positions:
            raise ValueError(f"No eligible positions for {name}.")
        weight_frame = pd.concat(positions).pivot(index="Date", columns="SecuritiesCode", values="Weight")
        weight_frame = weight_frame.reindex(sessions).fillna(0)
        daily = pd.DataFrame(stats, columns=[
            "Date", "Securities", "Leg size", "Median liquidity", "Missing selected labels", "Gross bps",
            "Top bps", "Bottom bps", "Market bps",
        ]).set_index("Date").reindex(sessions)
        inactive = daily["Leg size"].isna()
        daily.loc[inactive, ["Securities", "Leg size", "Missing selected labels", "Gross bps"]] = 0
        daily["Traded notional"] = portfolio_turnover(weight_frame)
        results.append(daily)
    return pd.concat(results, keys=[name for name, _ in books], names=["Book", "Date"])


def cost_sensitivity(daily: pd.DataFrame, rates_bps) -> pd.DataFrame:
    """Calculate net spread under one-way costs quoted per unit of traded notional."""
    rates = pd.Index(rates_bps, name="Cost rate bps")
    gross = daily["Gross bps"].mean(skipna=False)
    costs = rates.to_numpy() * daily["Traded notional"].mean()
    return pd.DataFrame({"Cost bps": costs, "Net bps": gross - costs}, index=rates)
