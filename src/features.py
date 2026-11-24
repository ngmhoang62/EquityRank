"""Construct causal predictors on the complete JPX session calendar."""

import numpy as np
import pandas as pd

from .preprocessing import KEY, eligible_bars


FEATURE_COLUMNS = (
    "return_1", "return_5", "return_20", "intraday_range",
    "volatility_20", "relative_turnover_20",
)
TREND_FEATURE_COLUMNS = ("return_60", "sma_gap_20")
SESSION_FEATURE_COLUMNS = ("overnight_gap", "intraday_return")
MARKET_FEATURE_COLUMNS = ("market_relative_1", "return_1_x_breadth", "return_20_x_dispersion")


def rank_features_by_session(features: pd.DataFrame) -> pd.DataFrame:
    """Rank available values within each signal session, before joining labels."""
    percentiles = features.groupby("Date", sort=False)[list(FEATURE_COLUMNS)].rank(
        method="average", pct=True, na_option="keep",
    )
    return pd.concat([features[KEY], percentiles], axis=1)


def build_price_features(
    prices: pd.DataFrame, *, include_liquidity: bool = False, include_trend: bool = False,
    include_context: bool = False,
    include_extended_trend: bool = False,
) -> pd.DataFrame:
    """Compute features as of the close of each signal session."""
    bars = prices.set_index(KEY)
    valid_bar = eligible_bars(bars)
    close = bars["Close"].where(valid_bar).unstack().sort_index()
    factor = bars["AdjustmentFactor"].unstack().reindex_like(close)
    past_log_factor = np.log(factor.where(np.isfinite(factor) & factor.gt(0))).shift(1)

    returns = {}
    windows = (1, 5, 20, 60) if include_trend else (1, 5, 20)
    if include_extended_trend:
        windows = (1, 5, 20, 40, 60)
    for window in windows:
        factor_product = np.exp(past_log_factor.rolling(window, min_periods=window).sum())
        returns[window] = close.div(close.shift(window).mul(factor_product)).sub(1)

    value_traded = (bars["Close"] * bars["Volume"]).where(valid_bar).unstack().reindex_like(close)
    past_median = value_traded.shift(1).rolling(20, min_periods=20).median()

    features = pd.DataFrame(index=bars.index)
    for window, values in returns.items():
        features[f"return_{window}"] = values.stack(future_stack=True).reindex(bars.index)
    features["intraday_range"] = ((bars["High"] - bars["Low"]) / bars["Open"]).where(valid_bar)
    features["volatility_20"] = (
        returns[1].rolling(20, min_periods=20).std().stack(future_stack=True).reindex(bars.index)
    )
    features["relative_turnover_20"] = (
        value_traded.div(past_median).stack(future_stack=True).reindex(bars.index)
    )
    if include_liquidity:
        features["Liquidity"] = past_median.stack(future_stack=True).reindex(bars.index)
    if include_trend or include_extended_trend:
        # Put closes on a shared scale using only factors effective through each session.
        # An unknown factor outside the 20-close window cancels from the ratio.
        adjusted_close = close.div(np.exp(past_log_factor.fillna(0).cumsum()))
        for window in ((20, 40, 60) if include_extended_trend else (20,)):
            complete_factors = past_log_factor.rolling(window - 1, min_periods=1).count().eq(window - 1)
            sma = adjusted_close.rolling(window, min_periods=window).mean().where(complete_factors)
            features[f"sma_gap_{window}"] = adjusted_close.div(sma).sub(1).stack(future_stack=True).reindex(bars.index)
        if include_extended_trend:
            for window in (40, 60):
                volatility = returns[1].rolling(window, min_periods=window).std()
                features[f"volatility_{window}"] = volatility.stack(future_stack=True).reindex(bars.index)
    if include_context:
        returns_20 = returns[1].rolling(20, min_periods=15)
        features["return_coverage_20"] = returns[1].notna().rolling(20, min_periods=1).sum().stack(future_stack=True)
        features["volatility_partial_20"] = returns_20.std().stack(future_stack=True)
        previous_close = close.shift(1).mul(np.exp(past_log_factor))
        open_price = bars["Open"].where(valid_bar)
        features["overnight_gap"] = open_price.div(previous_close.stack(future_stack=True)).sub(1)
        features["intraday_return"] = bars["Close"].where(valid_bar).div(open_price).sub(1)

        # Market aggregates use the full available price universe, before any label filtering.
        daily_return = returns[1].replace([np.inf, -np.inf], np.nan)
        mean_return = daily_return.mean(axis=1)
        breadth = np.sign(daily_return).mean(axis=1)
        dispersion = daily_return.std(axis=1, ddof=0)
        features["market_relative_1"] = daily_return.sub(mean_return, axis=0).stack(future_stack=True)
        features["return_1_x_breadth"] = daily_return.mul(breadth, axis=0).stack(future_stack=True)
        features["return_20_x_dispersion"] = returns[20].mul(dispersion, axis=0).stack(future_stack=True)
        features["market_dispersion"] = bars.index.get_level_values("Date").map(dispersion)
        paired_market = daily_return.notna().mul(mean_return, axis=0).where(daily_return.notna())
        market_variance = paired_market.rolling(60, min_periods=45).var()
        beta = daily_return.rolling(60, min_periods=45).cov(mean_return).div(market_variance)
        features["beta_60"] = beta.stack(future_stack=True)
    return features.replace([np.inf, -np.inf], np.nan).reset_index()
