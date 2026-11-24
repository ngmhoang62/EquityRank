"""Close-to-close marked positions with constant reference notional."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import rank_weights
from .preprocessing import KEY


def position_returns(prices):
    calendar = pd.DatetimeIndex(prices["Date"].unique()).sort_values()
    entry = pd.Series(calendar, index=calendar).shift(-1)
    exit_date = entry.shift(-1)
    signals = prices[KEY].assign(Entry=prices["Date"].map(entry), Exit=prices["Date"].map(exit_date))
    mark = prices[[*KEY, "Close", "AdjustmentFactor"]]
    signals = signals.merge(mark.rename(columns={"Date": "Entry", "Close": "Entry close"}),
                            on=["Entry", "SecuritiesCode"], how="left", validate="many_to_one")
    signals = signals.merge(mark[["Date", "SecuritiesCode", "Close"]].rename(
        columns={"Date": "Exit", "Close": "Exit close"}), on=["Exit", "SecuritiesCode"], how="left",
        validate="many_to_one")
    denominator = signals["Entry close"] * signals["AdjustmentFactor"]
    signals["Return"] = signals["Exit close"].where(signals["Exit close"].gt(0)).div(
        denominator.where(denominator.gt(0))).sub(1)
    return signals.set_index(KEY)["Return"]


def marked_book(ranks, prices, sessions, *, portfolio_size=200, returns=None):
    """Price selected ranks at t+1/t+2 and liquidate each continuously marked segment."""
    returns = position_returns(prices) if returns is None else returns
    count = ranks.groupby("Date")["Rank"].transform("size")
    eligible = count.ge(2 * portfolio_size)
    buy = ranks["Rank"].lt(portfolio_size)
    sell = ranks["Rank"].ge(count - portfolio_size)
    selected = ranks.loc[eligible & (buy | sell), [*KEY, "Rank"]].copy()
    weights = rank_weights(portfolio_size)
    long = buy.loc[selected.index].to_numpy()
    depth = np.where(long, selected["Rank"], count.loc[selected.index] - 1 - selected["Rank"])
    selected["Weight"] = weights[depth.astype(int)] * np.where(long, 1, -1)
    selected["Return"] = selected.set_index(KEY).index.map(returns)
    books = dict(tuple(selected.groupby("Date", sort=False)))
    full_calendar = pd.DatetimeIndex(prices["Date"].unique()).sort_values()
    index_on_calendar = pd.Series(np.arange(len(full_calendar)), index=full_calendar)
    rows = []
    previous = None
    previous_date = None
    segment = 0
    for date in sessions:
        day = books.get(date)
        priced = day is not None and np.isfinite(day["Return"]).all()
        consecutive = (
            previous_date is not None and index_on_calendar[date] == index_on_calendar[previous_date] + 1
        )
        if previous is not None and (not priced or not consecutive):
            rows[-1][3] += previous.abs().sum()
            previous = None
        if not priced:
            missing = 0 if day is None else int((~np.isfinite(day["Return"])).sum())
            rows.append([date, np.nan, np.nan, np.nan, np.nan, missing])
            previous_date = date
            continue
        if previous is None:
            segment += 1
        current = day.set_index("SecuritiesCode")["Weight"]
        realized = day.set_index("SecuritiesCode")["Return"]
        turnover = current.abs().sum() if previous is None else current.sub(previous, fill_value=0.).abs().sum()
        rows.append([date, segment, float(current @ realized), turnover, -current.clip(upper=0).sum(), 0])
        previous = current * (1 + realized)
        previous_date = date
    if previous is not None:
        rows[-1][3] += previous.abs().sum()
    return pd.DataFrame(rows, columns=["Date", "Segment", "Gross", "Turnover", "Short notional", "Missing marks"])


def cost_grid(book):
    rows = []
    for fee in (0, 2, 5, 10):
        for borrow in (0, 100, 300):
            net_bps = book["Gross"] * 10_000 - fee * book["Turnover"] - borrow * book["Short notional"] / 252
            equity = net_bps.groupby(book["Segment"]).cumsum()
            peak = equity.groupby(book["Segment"]).cummax().clip(lower=0)
            rows.append((fee, borrow, net_bps.mean(), (peak - equity).max(), int(net_bps.count())))
    return pd.DataFrame(rows, columns=["Fee bps", "Borrow bps/year", "Net bps", "Drawdown bps", "Priced sessions"])


@dataclass
class Position:
    order: int
    weight: float
    entry_price: float
    planned_exit: pd.Timestamp
    last_value: float


def simulate_execution(ranks, prices, sessions, *, portfolio_size=200):
    """Simulate close fills and retain inventories with an unobservable exit."""
    calendar = pd.DatetimeIndex(prices["Date"].unique()).sort_values()
    next_date = pd.Series(calendar, index=calendar).shift(-1)
    exit_date = next_date.shift(-1)
    sessions = pd.DatetimeIndex(sessions).sort_values()
    if sessions.empty or not sessions.is_unique or not sessions.isin(calendar).all():
        raise ValueError("Execution needs a unique nonempty signal calendar.")
    ranks = ranks.loc[ranks["Date"].isin(sessions)].sort_values(["Date", "Rank"])
    counts = ranks.groupby("Date")["Rank"].transform("size")
    if ranks.duplicated(KEY).any() or not ranks["Rank"].eq(ranks.groupby("Date").cumcount()).all():
        raise ValueError("Expected unique keys and consecutive ranks before execution.")
    long = ranks["Rank"].lt(portfolio_size)
    short = ranks["Rank"].ge(counts - portfolio_size)
    trade = counts.ge(2 * portfolio_size) & (long | short)
    orders = ranks.loc[trade, [*KEY, "Rank"]].reset_index(drop=True)
    weights = rank_weights(portfolio_size)
    depth = np.where(long.loc[trade], ranks.loc[trade, "Rank"], counts.loc[trade] - 1 - ranks.loc[trade, "Rank"])
    orders["Weight"] = weights[depth.astype(int)] * np.where(long.loc[trade], 1., -1.)
    orders["Entry"] = orders["Date"].map(next_date)
    orders["Planned exit"] = orders["Date"].map(exit_date)
    if exit_date.reindex(sessions).isna().any():
        raise ValueError("Execution requires a scheduled t+2 for each signal session.")
    observed_dates = calendar[(calendar >= next_date.loc[sessions[0]]) & (calendar <= exit_date.loc[sessions[-1]])]
    marks = prices.loc[prices["Date"].isin(observed_dates) & prices["SecuritiesCode"].isin(orders["SecuritiesCode"])]
    factor = marks.pivot(index="Date", columns="SecuritiesCode", values="AdjustmentFactor").reindex(observed_dates)
    # On absent records the dataset supplies no evidence of a corporate action.
    valid_factor = factor.where(np.isfinite(factor) & factor.gt(0), 1)
    scale = valid_factor.shift(1, fill_value=1).cumprod()
    valid_close = np.isfinite(marks["Close"]) & marks["Close"].gt(0) & marks["Volume"].gt(0)
    closes = marks.loc[valid_close].pivot(index="Date", columns="SecuritiesCode", values="Close")
    adjusted = closes.reindex_like(scale).div(scale)
    filled_weights = np.zeros(len(orders))
    exits = np.full(len(orders), np.datetime64("NaT"), dtype="datetime64[ns]")
    profits = np.full(len(orders), np.nan)
    status = np.full(len(orders), "unfilled", dtype=object)
    scheduled = {date: group for date, group in orders.groupby("Entry")}
    held = {}
    rows = []
    for date, marks_today in adjusted.iterrows():
        transactions = defaultdict(float)
        gross = borrow = 0.
        observed_positions = closed = deferred = 0
        for code, position in list(held.items()):
            price = marks_today[code]
            priced = np.isfinite(price)
            observed_positions += int(priced)
            if priced:
                position.last_value = position.weight * price / position.entry_price
            borrow += max(-position.last_value, 0.)
            if date < position.planned_exit or not priced:
                deferred += int(date == position.planned_exit and not priced)
                continue
            change = position.last_value - position.weight
            gross += change
            transactions[code] -= position.last_value
            exits[position.order] = date.to_datetime64()
            profits[position.order] = change
            status[position.order] = "closed"
            del held[code]
            closed += 1
        pending = {side: sum(abs(p.weight) for p in held.values() if p.weight * side > 0) for side in (1, -1)}
        selected = scheduled.get(date)
        fills = unfilled = blocked = 0
        if selected is not None:
            for order, code, weight, deadline in selected[["SecuritiesCode", "Weight", "Planned exit"]].itertuples():
                price = marks_today[code]
                capacity = max(0., 1. - pending[1 if weight > 0 else -1])
                if code in held or capacity == 0.:
                    status[order] = "blocked"
                    blocked += 1
                elif np.isfinite(price):
                    filled = weight * capacity
                    held[code] = Position(order, filled, price, deadline, filled)
                    transactions[code] += filled
                    filled_weights[order] = filled
                    status[order] = "open"
                    fills += 1
                else:
                    unfilled += 1
        exposure = {side: sum(abs(p.weight) for p in held.values() if p.weight * side > 0) for side in (1, -1)}
        delayed = sum(abs(p.weight) for p in held.values() if p.planned_exit <= date)
        rows.append((date, gross, sum(map(abs, transactions.values())), borrow, fills, unfilled, blocked, closed,
                     deferred, observed_positions, delayed, len(held), max(0., 1. - exposure[1]),
                     max(0., 1. - exposure[-1])))
    return pd.DataFrame(rows, columns=[
        "Date", "Gross", "Turnover", "Borrow notional", "Filled", "Unfilled", "Blocked", "Closed",
        "Deferred today", "Observed existing marks", "Deferred notional", "Open positions", "Long cash", "Short cash",
    ]), orders.assign(**{"Filled weight": filled_weights, "Exit": exits, "Gross": profits, "Status": status})


def execution_cost_grid(daily):
    result = []
    for fee in (0, 2, 5, 10):
        for borrow in (0, 100, 300):
            recognized = daily["Gross"] * 10_000 - fee * daily["Turnover"] - borrow * daily["Borrow notional"] / 252
            result.append((fee, borrow, recognized.sum(), recognized.mean(), len(daily)))
    return pd.DataFrame(result, columns=["Fee bps", "Borrow bps/year", "Recognized total bps",
                                         "Recognized bps per session", "Execution sessions"])
