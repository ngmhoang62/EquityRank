"""Reconstruct labels offline to check the effect of price precision and anchor dates."""

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


COLUMNS = ["Date", "SecuritiesCode", "Close", "AdjustmentFactor", "Target"]
LAST_TRAIN_DATE = "2021-12-03"


def rounded_close(value):
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def label_errors(prices):
    ordered = prices.sort_values(["SecuritiesCode", "Date"], ascending=[True, False])
    sessions = pd.Index(ordered["Date"].unique()).sort_values()
    session_id = pd.Series(sessions.get_indexer(ordered["Date"]), index=ordered.index)
    codes = ordered["SecuritiesCode"]
    by_code = session_id.groupby(codes, sort=False)
    consecutive = by_code.shift(1).eq(session_id + 1) & by_code.shift(2).eq(session_id + 2)
    scale = ordered.groupby("SecuritiesCode", sort=False)["AdjustmentFactor"].cumprod()
    adjusted = (ordered["Close"] * scale).map(rounded_close, na_action="ignore")
    grouped_close = adjusted.groupby(codes, sort=False)
    entry, exit_ = grouped_close.shift(1), grouped_close.shift(2)
    valid = consecutive & entry.gt(0) & exit_.gt(0) & ordered["Target"].notna()
    return (exit_[valid] / entry[valid] - 1 - ordered.loc[valid, "Target"]).abs()


def print_matches(label, errors):
    print(f"{label}: {errors.le(1e-8).sum():,} / {len(errors):,}; max abs error {errors.max():.3g}")


def main():
    raw = Path("data/raw")
    train = pd.read_csv(raw / "train_files/stock_prices.csv", usecols=COLUMNS)
    supplemental = pd.read_csv(raw / "supplemental_files/stock_prices.csv", usecols=COLUMNS)
    print_matches("Train-only adjustment", label_errors(train))
    combined = pd.concat([train, supplemental], ignore_index=True)
    errors = label_errors(combined)
    dates = combined.loc[errors.index, "Date"]
    print_matches("Combined adjustment, train", errors[dates.le(LAST_TRAIN_DATE)])
    print_matches("Combined adjustment, last two train sessions", errors[dates.isin(["2021-12-02", LAST_TRAIN_DATE])])
    print_matches("Combined adjustment, supplemental", errors[dates.gt(LAST_TRAIN_DATE)])


if __name__ == "__main__":
    main()
