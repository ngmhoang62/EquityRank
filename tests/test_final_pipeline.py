import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_pipeline import (
    load_config, fit_final_model, rank_prices, save_model, load_model, evaluate_predictions,
    paired_portfolio_summary,
)


class FinalPipelineTest(unittest.TestCase):
    def test_paired_summary_uses_the_same_priced_days_without_recomputing_turnover(self):
        dates = pd.date_range("2021-12-09", periods=3)
        daily = pd.DataFrame({
            "Date": dates.tolist() * 2,
            "Model": ["Ridge"] * 3 + ["Random 7"] * 3,
            "Gross bps": [10., 300., 6., 2., np.nan, 4.],
            "Traded notional": [1., 2., 3., 2., 4., 4.],
            "Leg size": [1] * 6,
            "IC": [.1] * 6,
        })
        matched, summary, costs = paired_portfolio_summary(daily, [0, 2])
        self.assertTrue(pd.DatetimeIndex(matched["Date"].unique()).equals(dates[[0, 2]]))
        self.assertAlmostEqual(summary.loc["Ridge", "Gross bps"], 8)
        self.assertAlmostEqual(costs.loc[("Ridge", 2), "Net bps"], 4)

    def test_fit_boundary_and_prediction_roundtrip_without_targets(self):
        dates = pd.bdate_range("2021-10-01", "2021-12-10")
        rows = pd.DataFrame(
            [(d, code) for d in dates for code in (1, 2, 3, 4)],
            columns=["Date", "SecuritiesCode"],
        )
        ticks = np.repeat(np.arange(len(dates)), 4)
        close = 100.0 + ticks * rows["SecuritiesCode"] / 10
        for column in ("Open", "High", "Low", "Close"):
            rows[column] = close
        rows["Volume"] = 1000
        rows["AdjustmentFactor"] = 1.0
        rows["Target"] = rows["SecuritiesCode"] * 0.01
        rows["ExpectedDividend"] = np.nan
        rows["SupervisionFlag"] = False

        config = load_config(Path(__file__).parents[1] / "configs/final_ridge.json")
        bundle = fit_final_model(rows, config)
        self.assertEqual(bundle["metadata"]["last_label_end"], "2021-12-08")
        changed = rows.copy()
        changed.loc[changed["Date"].ge("2021-12-07"), "Target"] = 1000
        inputs = changed.drop(columns="Target")
        expected = rank_prices(bundle, inputs, "2021-12-09")
        self.assertEqual(len(expected), 8)
        self.assertEqual(expected.groupby("Date")["Rank"].apply(list).tolist(), [[0, 1, 2, 3]] * 2)
        pd.testing.assert_frame_equal(expected, rank_prices(fit_final_model(changed, config), inputs, "2021-12-09"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ridge.joblib"
            save_model(bundle, path)
            pd.testing.assert_frame_equal(expected, rank_prices(load_model(path), inputs, "2021-12-09"))

    def test_missing_target_in_selected_leg_does_not_promote_another_security(self):
        date = pd.Timestamp("2021-12-09")
        predictions = pd.DataFrame({
            "Date": [date] * 4, "SecuritiesCode": [1, 2, 3, 4],
            "Score": [4., 3., 2., 1.], "Rank": [0, 1, 2, 3],
        })
        labels = predictions[["Date", "SecuritiesCode"]].assign(Target=[np.nan, .03, .01, -.01])
        config = {"portfolio_size": 1, "random_seeds": [7], "cost_rates_bps": [0, 5]}
        daily, summary, costs = evaluate_predictions(predictions, labels, pd.DatetimeIndex([date], name="Date"), config)
        ridge = daily.loc[daily["Model"].eq("Ridge")].iloc[0]
        self.assertEqual(ridge["Missing selected labels"], 1)
        self.assertTrue(pd.isna(ridge["Gross bps"]))
        self.assertTrue(pd.isna(ridge["Top bps"]))
        self.assertAlmostEqual(ridge["Bottom bps"], -100)
        self.assertAlmostEqual(ridge["Market bps"], 100)
        self.assertEqual(summary.loc["Ridge", "Known labels"], 3)
        self.assertTrue(costs.loc["Ridge", "Net bps"].isna().all())
