"""Price and point-in-time disclosure inputs for the release."""

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_price_features
from .preprocessing import KEY, OHLC, eligible_bars, load_stock_prices


PRICE_COLS = tuple([*OHLC, "Volume"])
RATIOS = ("operating_margin", "equity_asset_ratio")
REPORT_AGES = ("margin_report_age", "asset_report_age")
REPORT_MISSING = ("margin_report_missing", "asset_report_missing")
FIN_EVENTS = ("statement_today", "forecast_revision_today", "correction_today")
FIN_COLS = (*RATIOS, *REPORT_AGES, *REPORT_MISSING, *FIN_EVENTS)
LEGACY_FIN = ("legacy_operating_margin", "legacy_equity_asset_ratio", "legacy_report_age")
TREND_COLS = ("return_40", "volatility_40", "sma_gap_40", "return_60", "volatility_60", "sma_gap_60")
DISCLOSURE_COLS = [
    "SecuritiesCode", "DisclosedDate", "DisclosedTime", "DisclosureNumber",
    "DisclosedUnixTime", "TypeOfDocument", "CurrentPeriodEndDate", "TypeOfCurrentPeriod",
    "OperatingProfit", "NetSales", "Equity", "TotalAssets",
]


def read_release_inputs(raw: Path):
    folders = [raw / group for group in ("train_files", "supplemental_files")]
    train, supplement = [load_stock_prices(folder / "stock_prices.csv") for folder in folders]
    if train["Date"].max() >= supplement["Date"].min() or not train.columns.equals(supplement.columns):
        raise ValueError("Incompatible source price ranges or schemas.")
    prices = pd.concat([train, supplement], ignore_index=True)
    disclosures = pd.concat([
        pd.read_csv(folder / "financials.csv", usecols=DISCLOSURE_COLS, dtype="string")
        for folder in folders
    ], ignore_index=True)
    return prices, disclosures


def _disclosed_ratios(updates):
    for numerator, denominator, ratio in (("OperatingProfit", "NetSales", RATIOS[0]),
                                          ("Equity", "TotalAssets", RATIOS[1])):
        top = pd.to_numeric(updates[numerator], errors="coerce").astype(float)
        bottom = pd.to_numeric(updates[denominator], errors="coerce").astype(float)
        updates[ratio] = top.where(np.isfinite(top)).div(bottom.where(np.isfinite(bottom) & bottom.gt(0)))
        updates[ratio] = updates[ratio].where(np.isfinite(updates[ratio]))
    return updates


def _stock_codes(values):
    codes = pd.to_numeric(values, errors="coerce").astype(float)
    return codes.where(np.isfinite(codes) & codes.mod(1).eq(0))


def _asof(keys, updates, columns):
    joined = pd.merge_asof(
        keys[KEY].assign(row=np.arange(len(keys))).sort_values("Date", kind="stable"),
        updates[["SecuritiesCode", "effective", *columns]],
        left_on="Date", right_on="effective", by="SecuritiesCode", direction="backward",
    ).sort_values("row")
    joined.index = keys.index
    return joined


def prepare_disclosures(disclosures, calendar, code_dtype):
    """Classify documents and validate corrections against statements already published."""
    updates = disclosures.reset_index(drop=True).copy()
    updates["SecuritiesCode"] = _stock_codes(updates["SecuritiesCode"])
    updates["disclosed"] = pd.to_datetime(updates["DisclosedDate"], errors="coerce").dt.normalize()
    clock = pd.to_timedelta(updates["DisclosedTime"], errors="coerce")
    updates["stamp"] = updates["disclosed"] + clock.where(clock.ge(pd.Timedelta(0)) & clock.lt(pd.Timedelta(days=1)))
    if "DisclosedUnixTime" in updates:
        timestamp = pd.to_datetime(pd.to_numeric(updates["DisclosedUnixTime"], errors="coerce"),
                                   unit="s", utc=True, errors="coerce")
        local = timestamp.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None).astype("datetime64[ns]")
        updates["stamp"] = updates["stamp"].fillna(local.where(local.dt.normalize().eq(updates["disclosed"])))
    input_rows = len(updates)
    updates = updates.dropna(subset=["disclosed", "stamp", "SecuritiesCode", "DisclosureNumber"])
    valid_time_rows = len(updates)
    updates = updates.drop_duplicates().sort_values("stamp", kind="stable")
    exact_duplicates = valid_time_rows - len(updates)
    # A conflict arriving later must not alter a ratio already observable.
    identity = ["SecuritiesCode", "DisclosureNumber"]
    first = updates.groupby(identity)["stamp"].transform("min")
    same_time_conflict = updates.duplicated([*identity, "stamp"], keep=False)
    keep = updates["stamp"].eq(first) & ~same_time_conflict
    conflicting_rows = int((~keep).sum())
    updates = updates.loc[keep].copy()
    locations = calendar.searchsorted(updates["disclosed"], side="right")
    updates = updates.loc[locations < len(calendar)].copy()
    updates["effective"] = calendar.take(locations[locations < len(calendar)])
    updates["SecuritiesCode"] = updates["SecuritiesCode"].astype(code_dtype)
    updates["DisclosureNumber"] = updates["DisclosureNumber"].astype(str)
    updates["period_end"] = pd.to_datetime(updates["CurrentPeriodEndDate"], errors="coerce").dt.normalize()
    updates["period_type"] = updates["TypeOfCurrentPeriod"].astype("string")
    kind = updates["TypeOfDocument"].astype("string")
    statements = kind.str.contains("FinancialStatements", na=False)
    forecast = kind.str.startswith("ForecastRevision", na=False)
    correction = kind.str.startswith("NumericalCorrection", na=False)
    updates["scope"] = kind.str.extract(r"FinancialStatements_(.+)$", expand=False).where(statements)
    updates.loc[statements & updates["scope"].isna(), "scope"] = "unspecified"
    updates["statement_today"] = statements.astype("int8")
    updates["forecast_revision_today"] = forecast.astype("int8")
    updates["correction_today"] = correction.astype("int8")
    key = ["SecuritiesCode", "period_end", "period_type"]
    valid_period = updates["period_end"].le(updates["disclosed"])
    source = updates.loc[statements & valid_period].dropna(subset=key).sort_values("stamp", kind="stable").copy()
    source["scope_count"] = (~source.duplicated([*key, "scope"])).groupby(
        [source[column] for column in key], sort=False
    ).cumsum()
    revisions = updates.loc[correction].dropna(subset=key).sort_values("stamp", kind="stable")
    if not revisions.empty and not source.empty:
        matched = pd.merge_asof(revisions[[*key, "stamp"]].assign(row=revisions.index),
                                source[[*key, "stamp", "scope", "scope_count"]],
                                on="stamp", by=key, allow_exact_matches=False)
        unambiguous = matched.loc[matched["scope_count"].eq(1)].set_index("row")
        updates.loc[unambiguous.index, "scope"] = unambiguous["scope"]
    accepted = updates["scope"].notna() & valid_period
    updates.loc[~accepted, "scope"] = pd.NA
    updates = _disclosed_ratios(updates).sort_values(["effective", "stamp", "DisclosureNumber"], kind="stable")
    quality = {"Input documents": input_rows, "Invalid identity or timestamp": input_rows - valid_time_rows,
               "Duplicate records": exact_duplicates, "Conflicting records": conflicting_rows,
               "Eligible corrections": int((updates["scope"].notna() & updates["correction_today"].eq(1)).sum()),
               "Correction events": int(updates["correction_today"].sum())}
    return updates, quality


def financial_features(keys, disclosures, calendar, *, include_source=False):
    updates, _ = prepare_disclosures(disclosures, calendar, keys["SecuritiesCode"].dtype)
    result = pd.DataFrame(index=keys.index)
    for ratio, age, missing in zip(RATIOS, REPORT_AGES, REPORT_MISSING):
        history = updates.loc[updates["scope"].notna() & updates[ratio].notna()].copy()
        # A correction to an older period cannot replace a later available statement.
        precedence = history["period_end"].astype("int64") // 86_400_000_000_000 * 2
        precedence += history["scope"].str.startswith("Consolidated").astype(int)
        best_so_far = precedence.groupby(history["SecuritiesCode"], sort=False).cummax()
        history = history.loc[precedence.eq(best_so_far)]
        matched = _asof(keys, history, [ratio, "disclosed", "period_end", "scope", "DisclosureNumber"])
        result[ratio] = matched[ratio]
        result[age] = (matched["Date"] - matched["disclosed"]).dt.days
        result[missing] = matched[ratio].isna().astype("int8")
        if include_source:
            for column in ("disclosed", "effective", "period_end", "scope", "DisclosureNumber"):
                result[f"{ratio}_{column}"] = matched[column]
    events = updates.groupby(["effective", "SecuritiesCode"])[list(FIN_EVENTS)].max()
    result[list(FIN_EVENTS)] = events.reindex(pd.MultiIndex.from_frame(keys[KEY]), fill_value=0).to_numpy()
    return result


def legacy_financial_features(keys, disclosures, calendar):
    updates = disclosures.copy()
    updates["SecuritiesCode"] = _stock_codes(updates["SecuritiesCode"])
    updates["disclosed"] = pd.to_datetime(updates["DisclosedDate"], errors="coerce").dt.normalize()
    updates = updates.dropna(subset=["disclosed", "SecuritiesCode"]).copy()
    locations = calendar.searchsorted(updates["disclosed"], side="right")
    usable = locations < len(calendar)
    updates = updates.loc[usable].copy()
    updates["effective"] = calendar.take(locations[usable])
    updates = _disclosed_ratios(updates)
    updates["time"] = pd.to_timedelta(updates["DisclosedTime"], errors="coerce").fillna(pd.Timedelta(0))
    updates["SecuritiesCode"] = updates["SecuritiesCode"].astype(keys["SecuritiesCode"].dtype)
    updates = updates.sort_values(["effective", "disclosed", "time", "DisclosureNumber"], kind="stable")
    joined = _asof(keys, updates, ["disclosed", *RATIOS])
    joined["report_age"] = (joined["Date"] - joined["disclosed"]).dt.days
    return joined[[*RATIOS, "report_age"]].set_axis(LEGACY_FIN, axis=1)


def build_panel(prices, disclosures, *, extended=True, legacy_finance=False):
    features = build_price_features(prices, include_context=extended, include_extended_trend=extended,
                                    include_liquidity=extended)
    eligible = eligible_bars(prices)
    panel = prices.loc[eligible, [*KEY, *PRICE_COLS]].merge(features, on=KEY, validate="one_to_one")
    for column in PRICE_COLS:
        panel[f"log_{column}"] = np.log(panel[column])
    if extended:
        columns = list(FEATURE_COLUMNS)
        percentiles = panel.groupby("Date", sort=False)[columns].rank(pct=True)
        panel = panel.join(percentiles.add_prefix("percentile_"))
        calendar = pd.DatetimeIndex(prices["Date"].unique()).sort_values()
        panel = panel.join(financial_features(panel, disclosures, calendar))
        if legacy_finance:
            panel = panel.join(legacy_financial_features(panel, disclosures, calendar))
    return panel, features
