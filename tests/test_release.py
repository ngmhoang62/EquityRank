import unittest

import numpy as np
import pandas as pd

from src.release_data import financial_features
from src.backtest import cost_grid, marked_book
from src.release_experiment import calendar_labels, train_before
from src.release_models import guard_scores


class ReleaseTest(unittest.TestCase):
    def test_disclosures_take_effect_after_publication(self):
        dates = pd.to_datetime(["2021-03-12", "2021-03-15", "2021-03-16"])
        panel = pd.DataFrame({"Date": dates, "SecuritiesCode": 1})
        disclosures = pd.DataFrame({
            "SecuritiesCode": [1, 1], "DisclosedDate": ["2021-03-12", "2021-03-15"],
            "DisclosedTime": ["15:00:00", "09:00:00"], "DisclosureNumber": [1, 2],
            "TypeOfDocument": ["2QFinancialStatements_Consolidated_JP"] * 2,
            "CurrentPeriodEndDate": ["2020-09-30", "2020-09-30"],
            "TypeOfCurrentPeriod": ["2Q", "2Q"],
            "NetSales": [10., 20.], "OperatingProfit": [2., 8.], "Equity": [5., 6.], "TotalAssets": [10., 10.],
        })
        actual = financial_features(panel, disclosures, dates)
        np.testing.assert_allclose(actual["operating_margin"], [np.nan, .2, .4], equal_nan=True)
        np.testing.assert_allclose(actual["margin_report_age"], [np.nan, 3., 1.], equal_nan=True)

    def test_turnover_marks_existing_positions_and_charges_close(self):
        dates = pd.date_range("2021-03-01", periods=4)
        ranks = pd.DataFrame({
            "Date": np.repeat(dates[:2], 2), "SecuritiesCode": [1, 2, 1, 2], "Rank": [0, 1, 0, 1],
        })
        prices = pd.DataFrame({
            "Date": np.repeat(dates, 2), "SecuritiesCode": [1, 2] * 4,
            "Close": [100., 100., 100., 100., 55., 100., 55., 100.], "AdjustmentFactor": [1., 1.] * 2 + [1., 1.] * 2,
        })
        prices.loc[2, "AdjustmentFactor"] = .5
        result = marked_book(ranks, prices, pd.DatetimeIndex(dates[:2]), portfolio_size=1)
        np.testing.assert_allclose(result["Gross"], [.1, 0.])
        np.testing.assert_allclose(result["Turnover"], [2., 2.1])
        fees = cost_grid(result).query("`Fee bps` == 5 and `Borrow bps/year` == 0")
        self.assertAlmostEqual(fees["Net bps"].iloc[0], (1000 - 5 * 4.1) / 2)
        prices.loc[4, "Close"] = np.nan
        missing = marked_book(ranks, prices, pd.DatetimeIndex(dates[:2]), portfolio_size=1)
        self.assertTrue(missing["Gross"].isna().all())

    def test_zero_short_penalty_preserves_ridge_order(self):
        rows = pd.DataFrame({"Date": pd.Timestamp("2020-01-01"), "SecuritiesCode": [4, 3, 2, 1, 5]})
        ridge = pd.Series([2., 2., 1., 0., -1.])
        risk = pd.Series([0., 0., 0., 0., 1.])
        self.assertEqual(guard_scores(rows, ridge, risk, penalty=0., size=1).tolist(), ridge.tolist())
        safe = guard_scores(rows, ridge, risk, penalty=1., size=1)
        self.assertEqual(rows.loc[safe.idxmax(), "SecuritiesCode"], 3)
        self.assertEqual(rows.loc[safe.idxmin(), "SecuritiesCode"], 1)

    def test_label_at_first_validation_session_is_excluded_from_fit(self):
        dates = pd.bdate_range("2021-11-29", periods=7)
        prices = pd.DataFrame({"Date": dates, "SecuritiesCode": 1, "Target": np.arange(len(dates))})
        _, labels = calendar_labels(prices)
        train = train_before(labels, dates[5])
        self.assertEqual(train["Date"].tolist(), list(dates[:3]))
        self.assertEqual(train["LabelEndDate"].max(), dates[4])
