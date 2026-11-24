"""Profile the local JPX competition files and emit an aggregate JSON report."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_FOLDERS = ("train_files", "supplemental_files", "example_test_files")
TABLES = ("stock_prices", "secondary_stock_prices", "financials", "options", "trades")
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
KEY = ["Date", "SecuritiesCode"]
REQUIRED_FILES = {
    *(f"{folder}/{table}.csv" for folder in PRICE_FOLDERS for table in TABLES),
    *(f"data_specifications/{name}_spec.csv" for name in (
        "stock_price", "stock_list", "stock_fin", "options", "trades"
    )),
    "stock_list.csv",
    "example_test_files/sample_submission.csv",
    "jpx_tokyo_market_prediction/__init__.py",
    "jpx_tokyo_market_prediction/competition.cpython-37m-x86_64-linux-gnu.so",
}


def inventory(raw_dir):
    files = {}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entry = {"bytes": path.stat().st_size, "sha256": digest}
        if path.suffix == ".csv":
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            date_col = next((col for col in ("Date", "EffectiveDate") if col in columns), None)
            selected = sorted({columns[0], date_col or columns[0]})
            rows, first, last = 0, None, None
            for chunk in pd.read_csv(path, usecols=selected, dtype="string", chunksize=250_000):
                rows += len(chunk)
                if date_col and (dates := chunk[date_col].dropna()).size:
                    first = min(first or dates.min(), dates.min())
                    last = max(last or dates.max(), dates.max())
            entry.update(rows=rows, columns=len(columns))
            if date_col:
                entry.update(date_column=date_col, first_date=first, last_date=last)
        files[path.relative_to(raw_dir).as_posix()] = entry
    return files


def profile_prices(prices):
    ohlc = prices[PRICE_COLUMNS]
    missing_ohlc = ohlc.isna()
    invalid_ohlc_order = ohlc["High"].lt(ohlc[["Open", "Close", "Low"]].max(axis=1)) | ohlc["Low"].gt(
        ohlc[["Open", "Close", "High"]].min(axis=1)
    )
    daily = prices.groupby("Date")["SecuritiesCode"].nunique()
    stats = {
        "rows": len(prices),
        "sessions": len(daily),
        "securities": int(prices["SecuritiesCode"].nunique()),
        "securities_per_session": [int(daily.min()), float(daily.median()), int(daily.max())],
        "duplicate_row_ids": int(prices["RowId"].duplicated().sum()),
        "duplicate_date_code_keys": int(prices.duplicated(KEY).sum()),
        "row_id_mismatches": int((prices["RowId"] != (
            prices["Date"].dt.strftime("%Y%m%d") + "_" + prices["SecuritiesCode"].astype(str)
        )).sum()),
        "missing": {key: int(value) for key, value in prices.isna().sum().items()},
        "partial_ohlc_rows": int((missing_ohlc.any(axis=1) & ~missing_ohlc.all(axis=1)).sum()),
        "nonpositive_ohlc_rows": int((ohlc <= 0).any(axis=1).sum()),
        "ohlc_order_violations": int(invalid_ohlc_order.sum()),
        "zero_volume_rows": int(prices["Volume"].eq(0).sum()),
        "zero_volume_with_missing_ohlc": int((prices["Volume"].eq(0) & missing_ohlc.all(axis=1)).sum()),
        "negative_volume_rows": int(prices["Volume"].lt(0).sum()),
        "nonunit_adjustment_rows": int(prices["AdjustmentFactor"].dropna().ne(1).sum()),
        "nonpositive_adjustment_rows": int(prices["AdjustmentFactor"].le(0).sum()),
        "supervised_rows": int(prices["SupervisionFlag"].eq(True).sum()),
    }
    if "Target" in prices:
        target = prices["Target"]
        stats["target"] = {
            "nonfinite_nonmissing": int((target.notna() & ~np.isfinite(target)).sum()),
            "min": float(target.min()),
            "max": float(target.max()),
            "abs_gt_20_pct": int(target.abs().gt(0.20).sum()),
            "abs_gt_50_pct": int(target.abs().gt(0.50).sum()),
        }
    return stats


def check_label_timing(prices):
    prices = prices.sort_values(["SecuritiesCode", "Date"])
    sessions = pd.Index(prices["Date"].unique()).sort_values()
    session_id = pd.Series(sessions.get_indexer(prices["Date"]), index=prices.index)
    groups = prices.groupby("SecuritiesCode", sort=False)
    by_code = session_id.groupby(prices["SecuritiesCode"], sort=False)
    consecutive = by_code.shift(-1).eq(session_id + 1) & by_code.shift(-2).eq(session_id + 2)
    close_1, close_2 = groups["Close"].shift(-1), groups["Close"].shift(-2)
    factor_1 = groups["AdjustmentFactor"].shift(-1)
    target = prices["Target"]
    valid = consecutive & close_1.gt(0) & close_2.gt(0) & factor_1.gt(0) & target.notna()
    raw_error = (close_2[valid] / close_1[valid] - 1 - target[valid]).abs()
    adjusted_error = (close_2[valid] / (close_1[valid] * factor_1[valid]) - 1 - target[valid]).abs()
    split = factor_1[valid].ne(1)
    return {
        "final_two_sessions": [date.date().isoformat() for date in sessions[-2:]],
        "populated_labels_in_final_two_sessions": int(target[session_id.ge(len(sessions) - 2)].count()),
        "labeled_rows_with_two_observed_successors_nonconsecutive": int(
            (target.notna() & by_code.shift(-2).notna() & ~consecutive).sum()
        ),
        "comparable_rows": int(valid.sum()),
        "raw_match_at_1e_8": int(raw_error.le(1e-8).sum()),
        "adjusted_match_at_1e_8": int(adjusted_error.le(1e-8).sum()),
        "split_event_comparable_rows": int(split.sum()),
        "split_event_raw_match_at_1e_8": int(raw_error[split].le(1e-8).sum()),
        "split_event_adjusted_match_at_1e_8": int(adjusted_error[split].le(1e-8).sum()),
        "adjusted_abs_error_p99": float(adjusted_error.quantile(0.99)),
        "adjusted_abs_error_max": float(adjusted_error.max()),
    }


def profile_raw(raw_dir):
    files = inventory(raw_dir)
    missing = REQUIRED_FILES.difference(files)
    if missing:
        raise ValueError(f"Missing competition files: {', '.join(sorted(missing))}")

    stock_list = pd.read_csv(raw_dir / "stock_list.csv", usecols=["SecuritiesCode", "Universe0"])
    listed = set(stock_list["SecuritiesCode"])
    universe = set(stock_list.loc[stock_list["Universe0"].eq(True), "SecuritiesCode"])
    price_stats, price_keys = {}, {}
    for folder in PRICE_FOLDERS:
        for table in ("stock_prices", "secondary_stock_prices"):
            relative = f"{folder}/{table}.csv"
            prices = pd.read_csv(raw_dir / relative, parse_dates=["Date"])
            if prices[KEY].isna().any().any() or not pd.api.types.is_datetime64_any_dtype(prices["Date"]):
                raise ValueError(f"Missing or invalid price keys in {relative}")
            if ("Target" in prices) != (folder != "example_test_files"):
                raise ValueError(f"Unexpected Target schema in {relative}")
            codes = set(prices["SecuritiesCode"])
            price_stats[relative] = profile_prices(prices)
            price_stats[relative].update(
                codes_outside_stock_list=len(codes - listed), codes_in_universe0=len(codes & universe)
            )
            if table == "stock_prices":
                price_keys[folder] = pd.MultiIndex.from_frame(prices[KEY])
                if "Target" in prices:
                    price_stats[relative]["label_timing"] = check_label_timing(prices)

    sample = pd.read_csv(raw_dir / "example_test_files" / "sample_submission.csv", parse_dates=["Date"])
    sample_keys = pd.MultiIndex.from_frame(sample[KEY])
    sample_sizes = sample.groupby("Date")["Rank"].agg(["size", "nunique", "min", "max"])
    integers = sample["Rank"].mod(1).eq(0).groupby(sample["Date"]).all()
    valid_ranks = (
        integers & sample_sizes["size"].eq(sample_sizes["nunique"])
        & sample_sizes["min"].eq(0) & sample_sizes["max"].eq(sample_sizes["size"] - 1)
    )
    example_keys = price_keys["example_test_files"]
    return {
        "files": files,
        "stock_list": {"rows": len(stock_list), "unique_codes": len(listed), "universe0_codes": len(universe)},
        "prices": price_stats,
        "main_price_overlap": {
            "train_and_supplemental": len(price_keys["train_files"].intersection(price_keys["supplemental_files"])),
            "supplemental_and_example": len(price_keys["supplemental_files"].intersection(example_keys)),
        },
        "sample_submission": {
            "sessions": len(sample_sizes),
            "invalid_rank_sessions": int((~valid_ranks).sum()),
            "duplicate_keys": int(sample_keys.duplicated().sum()),
            "example_keys_matched": len(example_keys.intersection(sample_keys)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(profile_raw(args.raw_dir), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
