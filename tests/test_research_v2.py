from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.backtest import simulate_execution
from src.diagnostics import calendar_blocks
from src.evaluation import evaluate_ranked_books, jpx_sharpe, rank_scores
from src.features import build_price_features
from src.release_data import financial_features
from src.release_delivery import save_rank_identity, verify_rank_identity
from src.release_experiment import select_winner, stable_random_scores
from src.release_models import fit_candidate, predict_candidate


class ResearchProtocolTest(unittest.TestCase):
    def test_saved_ranks_and_controls_cannot_be_mixed_with_another_run(self):
        selection = {"protocol": "equity-rank-research-v2", "fingerprint": {"source": "one"},
                     "name": "Ridge six", "params": {"alpha": 10_000}}
        with TemporaryDirectory() as folder:
            output = Path(folder)
            for filename in ("predictions.csv", "ridge_predictions.csv", "control_scores.csv"):
                (output / filename).write_text("one", encoding="utf-8")
            save_rank_identity(output, selection)
            verify_rank_identity(output, selection)
            with self.assertRaises(ValueError):
                verify_rank_identity(output, {**selection, "params": {"alpha": 100}})
            (output / "control_scores.csv").write_text("two", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_rank_identity(output, selection)

    def test_selection_uses_validation_sharpe_even_if_cost_is_missing_or_negative(self):
        table = pd.DataFrame({"Model": ["Ridge six", "LightGBM OHLCV", "Random 7"],
                              "JPX Sharpe": [.02, .08, .70], "Spread bps": [5., 3., 15.],
                              "Net at 5 bps": [100., np.nan, 50.]})
        self.assertEqual(select_winner(table, ["Ridge six", "LightGBM OHLCV"]), "LightGBM OHLCV")
        with self.assertRaises(ValueError):
            select_winner(table.assign(**{"JPX Sharpe": np.nan}), ["Ridge six", "LightGBM OHLCV"])

    def test_missing_session_reduces_coverage_without_bridging_it(self):
        dates = pd.bdate_range("2020-09-01", periods=23)
        prices = pd.DataFrame({"Date": dates, "SecuritiesCode": 10, "Volume": 1., "AdjustmentFactor": 1.})
        for name in ("Open", "High", "Low", "Close"):
            prices[name] = np.arange(len(prices)) + 100.
        prices.loc[18, "Volume"] = 0
        first = build_price_features(prices.iloc[:22], include_context=True)
        extended = build_price_features(prices, include_context=True)
        pd.testing.assert_frame_equal(first, extended.iloc[:22])
        self.assertEqual(extended.loc[21, "return_coverage_20"], 18)
        self.assertTrue(np.isnan(extended.loc[19, "return_1"]))
        self.assertTrue(np.isnan(extended.loc[21, "volatility_20"]))
        self.assertTrue(np.isfinite(extended.loc[21, "volatility_partial_20"]))

    def test_direct_ranker_fits_predicts_and_is_reusable_without_labels(self):
        dates = pd.bdate_range("2020-11-02", periods=6)
        fit = pd.DataFrame({"Date": np.repeat(dates[:5], 40), "SecuritiesCode": np.tile(np.arange(40), 5)})
        fit["Target"] = fit["SecuritiesCode"] / 1_000
        columns = ("return_1", "return_5", "return_20", "intraday_range", "volatility_20", "relative_turnover_20")
        for col in columns:
            fit[col] = fit["Target"]
        signals = fit.iloc[:40].drop(columns="Target").assign(Date=dates[-1])
        model = fit_candidate("LambdaRank two sides", fit, {"min_child_samples": 5, "n_estimators": 20}, None, "cpu")
        scores = predict_candidate(model, signals, None, "cpu", {}, 4)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores.iloc[-1], scores.iloc[0])
    def test_document_events_do_not_erase_actuals_and_corrections_start_after_release(self):
        dates = pd.bdate_range("2020-11-02", periods=6)
        keys = pd.DataFrame({"Date": dates, "SecuritiesCode": 1})
        disclosures = pd.DataFrame({
            "SecuritiesCode": ["1", "1.0", "1.0", "1"], "DisclosedDate": dates[[0, 1, 2, 4]],
            "DisclosedTime": "15:00:00", "DisclosureNumber": ["10", "11", "12", "13"],
            "TypeOfDocument": ["2QFinancialStatements_Consolidated_JP", "ForecastRevision",
                               "NumericalCorrection", "NumericalCorrection"],
            "CurrentPeriodEndDate": pd.to_datetime(["2020-09-30"] * 3 + ["2019-09-30"]),
            "TypeOfCurrentPeriod": "2Q", "NetSales": [10., np.nan, 10., 10.],
            "OperatingProfit": [2., np.nan, 3., 9.],
            "TotalAssets": [20., np.nan, np.nan, 20.], "Equity": [10., np.nan, np.nan, 1.],
        })
        actual = financial_features(keys, disclosures, dates)
        np.testing.assert_allclose(actual["operating_margin"], [np.nan, .2, .2, .3, .3, .3])
        np.testing.assert_allclose(actual["equity_asset_ratio"], [np.nan, .5, .5, .5, .5, .5])
        self.assertEqual(actual.loc[2, "forecast_revision_today"], 1)
        self.assertEqual(actual.loc[3, "asset_report_age"], 3)
        prefix = financial_features(keys.iloc[:3], disclosures.iloc[:2], dates[:3])
        pd.testing.assert_frame_equal(prefix, actual.iloc[:3])

    def test_rank_and_random_use_keys_even_when_row_order_changes(self):
        keys = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"] * 3), "SecuritiesCode": [12, 10, 11]})
        scores = pd.Series([.1, .1, -.2])
        ranked = rank_scores(keys, scores)
        self.assertEqual(ranked["SecuritiesCode"].tolist(), [10, 12, 11])
        self.assertEqual(ranked["Rank"].tolist(), [0, 1, 2])
        with self.assertRaises(ValueError):
            rank_scores(keys, pd.Series([.1, np.inf, -.2]))
        with self.assertRaises(ValueError):
            rank_scores(pd.concat([keys, keys], ignore_index=True), pd.concat([scores, scores], ignore_index=True))
        permuted = keys.iloc[::-1]
        pd.testing.assert_series_equal(stable_random_scores(keys, 7), stable_random_scores(permuted, 7).sort_index())

    def test_reference_weight_normalization_keeps_the_same_sharpe(self):
        keys = pd.DataFrame({"Date": np.repeat(pd.bdate_range("2020-11-02", periods=3), 4),
                             "SecuritiesCode": np.tile([1, 2, 3, 4], 3)})
        scores = pd.DataFrame({"A": np.tile([.4, .3, .2, .1], 3)})
        labels = keys.assign(Target=[.03, .02, -.01, -.02, -.01, .01, .02, .00, .02, .01, .01, .00])
        _, metrics = evaluate_ranked_books(keys, scores, labels, portfolio_size=2)
        # The reference divides total weighted return by mean weight, 1.5 here.
        reference = pd.Series([(2 * .03 + .02 + .01 + 2 * .02) / 1.5,
                               (2 * -.01 + .01 - .02 - 2 * .00) / 1.5,
                               (2 * .02 + .01 - .01 - 2 * .00) / 1.5])
        np.testing.assert_allclose(metrics["Spread"] * 2, reference)
        self.assertAlmostEqual(jpx_sharpe(metrics["Spread"]), reference.mean() / reference.std(ddof=1))

    def test_a_later_duplicate_document_does_not_change_past_ratios(self):
        dates = pd.bdate_range("2020-11-02", periods=4)
        keys = pd.DataFrame({"Date": dates, "SecuritiesCode": 1})
        raw = pd.DataFrame({
            "SecuritiesCode": ["1", "1.0"], "DisclosedDate": dates[[0, 2]],
            "DisclosedTime": "15:00:00", "DisclosureNumber": "7",
            "TypeOfDocument": "FYFinancialStatements_Consolidated_JP",
            "CurrentPeriodEndDate": "2020-09-30", "TypeOfCurrentPeriod": "FY",
            "NetSales": 10., "OperatingProfit": [2., 9.], "Equity": 5., "TotalAssets": 10.,
        })
        original = financial_features(keys, raw.iloc[:1], dates)
        merged = financial_features(keys, raw, dates)
        pd.testing.assert_frame_equal(original, merged)

    def test_bootstrap_does_not_join_dates_across_missing_sessions(self):
        intervals = calendar_blocks(np.array([.01, np.nan, .03, .04]), 2, 12, np.random.default_rng(7))
        self.assertEqual(intervals.shape, (12, 4))
        for left, right in intervals.reshape(-1, 2):
            self.assertTrue((np.isnan(left) and right == .03) or (left == .01 and np.isnan(right))
                            or (left == .03 and right == .04))

    def test_execution_misses_entry_defers_exit_and_accounts_for_split(self):
        dates = pd.bdate_range("2020-11-02", periods=5)
        prices = pd.DataFrame([(date, code) for date in dates for code in (1, 2)],
                              columns=["Date", "SecuritiesCode"])
        prices["Close"] = [100., 100., 100., np.nan, np.nan, 100., 55., 100., 60., 100.]
        prices["Volume"] = np.where(prices["Close"].notna(), 100, 0)
        prices["AdjustmentFactor"] = [1., 1., .5, 1., 1., 1., 1., 1., 1., 1.]
        ranks = pd.DataFrame({"Date": np.repeat(dates[[0, 1, 2]], 2),
                              "SecuritiesCode": [1, 2] * 3, "Rank": [0, 1] * 3})
        daily, orders = simulate_execution(ranks, prices, dates[:3], portfolio_size=1)
        first = orders.loc[orders["Date"].eq(dates[0])].set_index("SecuritiesCode")
        self.assertEqual(first.loc[1, "Exit"], dates[3])
        self.assertAlmostEqual(first.loc[1, "Gross"], .10)
        self.assertEqual(first.loc[2, "Status"], "unfilled")
        self.assertEqual(orders.loc[orders["Date"].eq(dates[1]) & orders["SecuritiesCode"].eq(1),
                                    "Status"].item(), "blocked")
        self.assertEqual(daily["Unfilled"].sum(), 1)
        self.assertEqual(daily["Blocked"].sum(), 1)
        self.assertEqual(daily["Long cash"].iloc[1], 0.)
        self.assertAlmostEqual(daily["Turnover"].sum(), 4.190909090909091)
