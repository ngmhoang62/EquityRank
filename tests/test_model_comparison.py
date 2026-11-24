import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.pipeline import make_pipeline

from src.baselines import regression_preprocessing, ridge_scores
from src.diagnostics import short_tail_diagnostics, spread_mean_interval
from src.evaluation import evaluate_scores
from src.features import (
    FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, SESSION_FEATURE_COLUMNS, TREND_FEATURE_COLUMNS,
    build_price_features, rank_features_by_session,
)
from src.models.lightgbm import lightgbm_rank_scores, lightgbm_scores
from src.research import inner_training_split


class ModelComparisonTest(unittest.TestCase):
    def test_short_tail_diagnostic_uses_same_ranks_weights_and_returns(self):
        frame = pd.DataFrame({
            "Date": pd.Timestamp("2021-06-04"), "SecuritiesCode": [1, 2, 3, 4],
            "Target": [0.10, -0.02, 0.01, 0.03],
        })
        scores = pd.DataFrame({"Model": [0., 1., 2., 3.]})
        daily = short_tail_diagnostics(frame, scores, portfolio_size=2, tail_size=1).iloc[0]
        self.assertAlmostEqual(daily["Short excess"], -(2 * 0.07 - 0.05) / 3)
        self.assertAlmostEqual(daily["Largest loss excess"], 2 * 0.07 / 3)
        self.assertAlmostEqual(daily["Share above market"], 0.5)

    def test_sequence_batches_preserve_missing_sessions_and_stop_at_signal(self):
        import torch
        from src.models.neural import history_batch

        panel = torch.tensor([[[1.], [101.]], [[0.], [102.]], [[3.], [103.]], [[4.], [104.]]])
        rows = torch.tensor([[2, 0], [1, 1]])
        result = history_batch(panel, rows, torch.arange(3))
        np.testing.assert_allclose(result.numpy(), [[[1., 0., 3.]], [[0., 101., 102.]]])

    def test_robust_preprocessing_uses_training_quantiles_for_evaluation(self):
        fit = pd.DataFrame({"signal": [0., 1., 2., 3., 4., np.nan]})
        preprocess = make_pipeline(*regression_preprocessing(robust=True)).fit(fit)
        evaluation = pd.DataFrame({"signal": [1e6, 2., np.nan]})
        values = preprocess.transform(evaluation)
        np.testing.assert_allclose(values[:, 0], [8., 0., 0.])
        self.assertGreater(values[2, 1], values[1, 1])

    def test_neural_scores_require_only_evaluation_features(self):
        from src.models.neural import neural_return_scores

        fit = pd.DataFrame({"feature": np.linspace(-1., 1., 36), "Target": np.linspace(-.01, .01, 36)})
        evaluation = pd.DataFrame({"feature": [np.nan, 0.25]}, index=[22, 6])
        scores = neural_return_scores(
            fit, evaluation, ("feature",), device="cpu", epochs=2, batch_size=8, include_huber=True,
        )
        self.assertEqual(scores.index.tolist(), [22, 6])
        self.assertEqual(scores.columns.tolist(), ["MLP", "MLP residual", "MLP Huber"])
        self.assertTrue(np.isfinite(scores.to_numpy()).all())
        self.assertFalse(np.allclose(scores["MLP"], scores["MLP Huber"]))

    def test_market_and_session_features_use_same_day_peers_and_adjusted_gap(self):
        dates = pd.bdate_range("2020-01-01", periods=22)
        prices = pd.DataFrame([
            (day, code, 100.0, 100.0, 100.0, 100.0, 10, 1.0)
            for day in dates for code in (11, 22)
        ], columns=["Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume", "AdjustmentFactor"])
        prices.loc[38, "AdjustmentFactor"] = 0.5
        prices.loc[40, ["Open", "High", "Low", "Close"]] = [50., 55., 50., 55.]
        features = build_price_features(prices, include_context=True)
        np.testing.assert_allclose(features.loc[40, list(SESSION_FEATURE_COLUMNS)].astype(float), [0., 0.1])
        np.testing.assert_allclose(features.loc[40, list(MARKET_FEATURE_COLUMNS)].astype(float), [0.05, 0.05, 0.1 * 0.05])
        subset = build_price_features(prices.iloc[:42], include_context=True)
        pd.testing.assert_frame_equal(features.iloc[:42], subset)

    def test_inner_search_purges_labels_crossing_into_inner_evaluation(self):
        dates = pd.bdate_range("2021-01-04", periods=10)
        frame = pd.DataFrame({"Date": dates[:-2], "LabelEndDate": dates[2:]})
        train, evaluation = inner_training_split(frame, sessions=3)
        self.assertEqual(train["Date"].tolist(), dates[:3].tolist())
        self.assertEqual(evaluation["Date"].tolist(), dates[5:8].tolist())

    def test_long_trend_features_use_split_adjusted_history_through_signal_date(self):
        dates = pd.bdate_range("2020-01-01", periods=75)
        adjusted = np.arange(100., 175.)
        close = adjusted.copy()
        close[55:] *= 0.5
        close[71:] *= 0.5
        prices = pd.DataFrame({column: close for column in ("Open", "High", "Low", "Close")}).assign(
            Date=dates, SecuritiesCode=11, Volume=10, AdjustmentFactor=1.0,
        )
        prices.loc[[54, 70], "AdjustmentFactor"] = 0.5
        point = 65
        features = build_price_features(prices, include_trend=True)

        base_features = build_price_features(prices)
        self.assertNotIn("return_60", base_features)
        pd.testing.assert_frame_equal(features[base_features.columns], base_features)
        self.assertTrue(pd.isna(features.loc[59, "return_60"]))
        self.assertTrue(pd.isna(features.loc[18, "sma_gap_20"]))
        self.assertAlmostEqual(features.loc[point, "return_60"], adjusted[point] / adjusted[point - 60] - 1)
        sma = adjusted[point - 19:point + 1].mean()
        self.assertAlmostEqual(features.loc[point, "sma_gap_20"], adjusted[point] / sma - 1)
        truncated = build_price_features(prices.iloc[:point + 1], include_trend=True)
        trend_columns = list(TREND_FEATURE_COLUMNS)
        pd.testing.assert_series_equal(features.loc[point, trend_columns], truncated.loc[point, trend_columns])

        prices.loc[60, "Volume"] = 0
        missing_bar = build_price_features(prices, include_trend=True)
        self.assertTrue(pd.isna(missing_bar.loc[point, "sma_gap_20"]))
        self.assertAlmostEqual(missing_bar.loc[point, "return_60"], features.loc[point, "return_60"])
        prices.loc[56, "AdjustmentFactor"] = np.nan
        missing_factor = build_price_features(prices, include_trend=True)
        self.assertTrue(pd.isna(missing_factor.loc[point, "return_60"]))

    def test_dual_ranker_groups_training_rows_and_scores_without_labels(self):
        dates = pd.date_range("2021-01-04", periods=4)
        values = np.tile(np.linspace(-1., 1., 600), len(dates))
        train = pd.DataFrame({column: values for column in FEATURE_COLUMNS}).assign(
            Date=np.repeat(dates, 600), SecuritiesCode=np.tile(np.arange(600), len(dates)), Target=values,
        )
        evaluation = train.loc[[0, 600, 599, 1199]].drop(columns="Target")
        shuffled = train.sample(frac=1, random_state=7)

        with patch.object(LGBMRanker, "fit", autospec=True, side_effect=LGBMRanker.fit) as fit:
            scores = lightgbm_rank_scores(shuffled, evaluation)

        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            np.testing.assert_array_equal(call.kwargs["group"], [600] * 4)
        long_labels = np.asarray(fit.call_args_list[0].args[2])
        short_labels = np.asarray(fit.call_args_list[1].args[2])
        np.testing.assert_array_equal(long_labels[[0, 599, 600, 1199]], [0, 2, 0, 2])
        np.testing.assert_array_equal(short_labels[[0, 599, 600, 1199]], [2, 0, 2, 0])
        self.assertEqual(scores.index.tolist(), [0, 600, 599, 1199])
        np.testing.assert_allclose(scores, [-0.5, -0.5, 0.5, 0.5])

    def test_ridge_target_percentiles_rank_within_training_days_only(self):
        train = pd.DataFrame({column: [1., 2., 3., 4., 5., 6.] for column in FEATURE_COLUMNS})
        train["Date"] = pd.to_datetime(["2021-01-04"] * 3 + ["2021-01-05"] * 3)
        train["Target"] = [0.02, 0.02, -0.01, -0.10, -0.06, -0.03]
        original = train.copy()
        evaluation = train.loc[[4, 0]].copy()
        evaluation["Target"] = [0.5, -0.8]
        expected_train = train.assign(Target=[5 / 6, 5 / 6, 1 / 3, 1 / 3, 2 / 3, 1.0])

        actual = ridge_scores(train, evaluation, target_percentiles=True)

        pd.testing.assert_series_equal(actual, ridge_scores(expected_train, evaluation))
        pd.testing.assert_series_equal(actual, ridge_scores(train, evaluation.drop(columns="Target"), target_percentiles=True))
        pd.testing.assert_frame_equal(train, original)
        self.assertFalse(np.allclose(actual, ridge_scores(train, evaluation)))

    def test_feature_ranks_use_same_day_peers_with_ties_and_missing_values(self):
        dates = pd.to_datetime(["2021-06-04", "2021-06-07", "2021-06-04", "2021-06-04"])
        frame = pd.DataFrame({"Date": dates, "SecuritiesCode": [11, 11, 22, 33]}, index=[8, 4, 7, 5])
        for column in FEATURE_COLUMNS:
            frame[column] = [0.01, -0.05, 0.01, 0.03]
        frame.loc[5, "return_5"] = np.nan
        frame["Target"] = [0.02, -0.01, np.nan, 0.50]

        ranked = rank_features_by_session(frame)

        pd.testing.assert_frame_equal(ranked[["Date", "SecuritiesCode"]], frame[["Date", "SecuritiesCode"]])
        self.assertNotIn("Target", ranked)
        np.testing.assert_allclose(ranked["return_1"], [0.5, 1.0, 0.5, 1.0])
        np.testing.assert_allclose(ranked["return_5"], [0.75, 1.0, 0.75, np.nan], equal_nan=True)

    def test_daily_comparison_exposes_both_return_legs(self):
        date = pd.Timestamp("2021-06-04")
        frame = pd.DataFrame({"Date": [date, date], "SecuritiesCode": [1, 2], "Target": [0.02, -0.01]})
        _, _, daily = evaluate_scores(frame, pd.DataFrame({"Model": [2., 1.]}), portfolio_size=1)
        np.testing.assert_allclose(daily.loc[0, ["Top", "Bottom", "Market"]].astype(float), [0.02, -0.01, 0.005])

    def test_interval_reports_mean_difference_in_bps(self):
        difference = pd.Series([0.0002] * 12)
        np.testing.assert_allclose(spread_mean_interval(difference), [2.0, 2.0])
        with self.assertRaises(ValueError):
            spread_mean_interval(difference, block_size=20)

    def test_lightgbm_uses_train_labels_and_keeps_evaluation_index(self):
        feature = np.linspace(-1.0, 1.0, 2400)
        train = pd.DataFrame({column: feature for column in FEATURE_COLUMNS})
        train["Target"] = np.where(feature > 0, 0.03, -0.01)
        evaluation = train.loc[[200, 1800], list(FEATURE_COLUMNS)].copy()
        evaluation.iloc[0, 1] = np.nan
        evaluation.index = [8, 4]
        scores = lightgbm_scores(train, evaluation)
        self.assertEqual(scores.index.tolist(), [8, 4])
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores.loc[4], scores.loc[8])
