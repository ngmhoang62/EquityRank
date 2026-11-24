"""Prepare the JPX price panel for supervised experiments."""

from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["Date", "SecuritiesCode"]
OHLC = ["Open", "High", "Low", "Close"]
MODEL_FEATURE_COLUMNS = (*OHLC, "Volume", "AdjustmentFactor")
TARGET = "Target"
LABEL_END = "LabelEndDate"


def eligible_bars(prices):
    values = prices[[*OHLC, "Volume"]]
    return np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)


def load_stock_prices(path: str | Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the combined panel and validate its chronological keys."""
    prices = pd.read_csv(path, usecols=columns, parse_dates=["Date"], float_precision="round_trip")
    if not pd.api.types.is_datetime64_any_dtype(prices["Date"]):
        raise ValueError("Date must contain valid trading dates.")
    if prices[KEY].isna().any().any() or prices.duplicated(KEY).any():
        raise ValueError("Expected one row per nonmissing (Date, SecuritiesCode).")
    return prices.sort_values(KEY, ignore_index=True)


def prepare_supervised(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filter unusable rows and attach the label's realization date from the full session calendar."""
    sessions = pd.DatetimeIndex(prices["Date"].unique()).sort_values()
    end_by_session = pd.Series(sessions, index=sessions).shift(-2)
    label_end = prices["Date"].map(end_by_session)

    has_bar = eligible_bars(prices)
    has_target = np.isfinite(prices[TARGET])
    known_label_end = label_end.notna()
    keep = has_bar & has_target & known_label_end
    prepared = prices.loc[keep].drop(columns=["ExpectedDividend", "SupervisionFlag"])
    prepared[LABEL_END] = label_end.loc[keep]
    audit = pd.Series({
        "Dòng đầu vào": len(prices),
        "Loại vì thiếu OHLC hoặc Volume không dương": int((~has_bar).sum()),
        "Loại tiếp vì thiếu Target": int((has_bar & ~has_target).sum()),
        "Loại tiếp vì t+2 nằm ngoài lịch quan sát": int((has_bar & has_target & ~known_label_end).sum()),
        "Dòng đủ điều kiện chia tập": len(prepared),
    }, name="Số dòng")
    return prepared.reset_index(drop=True), audit


def model_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the baseline predictors and label using an explicit feature list."""
    return frame.loc[:, list(MODEL_FEATURE_COLUMNS)], frame[TARGET]
