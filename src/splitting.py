"""Chronological splits with two-session label purging."""

import pandas as pd

from .preprocessing import LABEL_END


def development_windows(test_start: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Use consecutive half-year windows, ending at the historical holdout."""
    cutoff = pd.Timestamp(test_start)
    bounds = [*pd.date_range("2019-01-01", "2021-07-01", freq="6MS"), cutoff]
    if cutoff <= bounds[-2]:
        raise ValueError("The cutoff must follow the beginning of 2021 H2.")
    return list(zip(bounds[:-1], bounds[1:]))


def purged_window(frame: pd.DataFrame, start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fit and evaluation rows with all labels realized within their windows."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end:
        raise ValueError("Expected a positive evaluation window.")
    date, label_end = frame["Date"], frame[LABEL_END]
    return (
        frame.loc[date.lt(start) & label_end.lt(start)],
        frame.loc[date.ge(start) & date.lt(end) & label_end.lt(end)],
    )


def suggest_split_dates(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Aim for at least 80% fit rows and about 10% test rows using whole sessions."""
    signal_counts = frame.groupby("Date").size()
    session_dates = signal_counts.index
    cumulative_test = signal_counts.iloc[::-1].cumsum().iloc[::-1]
    test_start = (cumulative_test - 0.10 * len(frame)).abs().idxmin()

    labels_per_end = frame.groupby(LABEL_END).size().sort_index()
    end_counts = labels_per_end.cumsum()
    # Use the number of completed labels strictly before each candidate boundary.
    fit_counts = end_counts.reindex(end_counts.index.union(session_dates).sort_values()).ffill()
    fit_counts = fit_counts.shift(1).reindex(session_dates).fillna(0)
    valid_starts = fit_counts.loc[fit_counts.ge(0.80 * len(frame)) & (session_dates < test_start)]
    if valid_starts.empty:
        raise ValueError("Not enough sessions to find an 80/10/10 split after label purging.")
    return valid_starts.index[0], test_start


def split_by_date(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Assign complete sessions to periods, then purge labels crossing their boundary."""
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if not validation_start < test_start:
        raise ValueError("Validation must start before test.")

    date, label_end = frame["Date"], frame[LABEL_END]
    periods = {
        "train": (date.lt(validation_start), label_end.lt(validation_start)),
        "validation": (date.ge(validation_start) & date.lt(test_start), label_end.lt(test_start)),
        "test": (date.ge(test_start), label_end.notna()),
    }

    splits, counts = {}, []
    for name, (in_period, label_allowed) in periods.items():
        rows = frame.loc[in_period & label_allowed]
        if rows.empty:
            raise ValueError(f"No usable rows in the {name} split.")
        splits[name] = rows
        counts.append({
            "Tập": name,
            "Phiên đầu": rows["Date"].min().date(),
            "Phiên cuối": rows["Date"].max().date(),
            "Ngày nhãn hoàn tất muộn nhất": rows[LABEL_END].max().date(),
            "Số phiên giữ lại": rows["Date"].nunique(),
            "Số dòng bị loại ở ranh giới": int(in_period.sum()) - len(rows),
            "Số dòng giữ lại": len(rows),
        })
    summary = pd.DataFrame(counts).set_index("Tập")
    kept = summary["Số dòng giữ lại"]
    summary["Tỷ lệ (%)"] = (100 * kept / kept.sum()).round(2)
    return splits, summary
