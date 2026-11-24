import unittest

import numpy as np
import pandas as pd

from src.liquidity import (
    FULL_BOOK, cost_sensitivity, liquidity_terciles, matched_liquidity_portfolios, portfolio_turnover,
)


class LiquidityCostsTest(unittest.TestCase):
    def test_matched_books_use_the_same_daily_leg_sizes_and_known_liquidity(self):
        sessions = pd.DatetimeIndex(["2021-01-04", "2021-01-05"], name="Date")
        frame = pd.concat([
            pd.DataFrame({
                "Date": date, "SecuritiesCode": range(size + 1), "Score": range(size + 1),
                "Liquidity": [*range(1, size + 1), np.nan],
                "Target": [*np.arange(size) / 10_000, 0.5],
            }) for date, size in zip(sessions, (30, 60))
        ], ignore_index=True)
        daily = matched_liquidity_portfolios(frame, sessions)
        full, mid = daily.xs(FULL_BOOK), daily.xs("Vừa")
        self.assertEqual(full["Leg size"].tolist(), [1, 2])
        self.assertEqual(mid["Leg size"].tolist(), [1, 2])
        self.assertEqual(full["Securities"].tolist(), [30, 60])
        self.assertEqual(mid["Securities"].tolist(), [10, 20])
        self.assertAlmostEqual(full["Gross bps"].iloc[0], 29)
        self.assertAlmostEqual(mid["Gross bps"].iloc[0], 9)

    def test_rank_terciles_reset_each_session_and_keep_missing_out(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2021-01-04"] * 4 + ["2021-01-05"] * 3),
            "SecuritiesCode": [4, 3, 2, 1, 4, 3, 2],
            "Liquidity": [30., float("nan"), 10., 20., 10., 20., 30.],
        })
        groups = liquidity_terciles(frame)
        self.assertEqual(groups.dropna().tolist(), ["Cao", "Thấp", "Vừa", "Thấp", "Vừa", "Cao"])
        self.assertTrue(pd.isna(groups.iloc[1]))

    def test_turnover_counts_open_exit_and_crossing_from_long_to_short(self):
        weights = pd.DataFrame([[1., -1.], [1., -1.], [-1., 1.]])
        self.assertEqual(portfolio_turnover(weights).tolist(), [2., 0., 6.])

    def test_rate_applies_to_absolute_traded_notional(self):
        daily = pd.DataFrame({"Gross bps": [10., 6.], "Traded notional": [2., 1.]})
        summary = cost_sensitivity(daily, [0, 2, 5])
        np.testing.assert_allclose(summary["Net bps"], [8., 5., .5])

    def test_unknown_gross_does_not_disappear_from_cost_summary(self):
        daily = pd.DataFrame({"Gross bps": [10., np.nan], "Traded notional": [2., 1.]})
        self.assertTrue(cost_sensitivity(daily, [0, 2])["Net bps"].isna().all())
