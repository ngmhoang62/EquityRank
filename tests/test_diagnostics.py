import unittest

import pandas as pd

from src.diagnostics import rank_diagnostics


class RankDiagnosticsTest(unittest.TestCase):
    def test_uses_common_support_and_correct_tail_orientation(self):
        date = pd.Timestamp("2021-06-04")
        frame = pd.DataFrame({
            "Date": [date] * 5,
            "SecuritiesCode": [4, 2, 1, 3, 5],
            "Target": [-0.03, 0.02, 0.04, -0.01, 9.0],
        })
        scores = pd.DataFrame({"Ridge": [1., 3., 4., 2., 10.], "Other": [1., 1., 1., 1., float("nan")]})
        daily, groups, _ = rank_diagnostics(frame, scores, "Ridge", portfolio_size=2, n_groups=2, n_random=3)
        self.assertAlmostEqual(daily.loc[date, "Top"], 0.03333333333333333)
        self.assertAlmostEqual(daily.loc[date, "Bottom"], -0.02333333333333333)
        self.assertAlmostEqual(daily.loc[date, "Spread"], 0.05666666666666667)
        self.assertAlmostEqual(groups.loc[(date, 1), "Excess"], 0.025)
        self.assertAlmostEqual(groups.loc[(date, 2), "Excess"], -0.025)

    def test_random_groups_do_not_overlap(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2021-06-04"] * 2),
            "SecuritiesCode": [1, 2],
            "Target": [0.02, -0.01],
        })
        scores = pd.DataFrame({"Ridge": [1., 0.]})
        _, _, simulations = rank_diagnostics(frame, scores, "Ridge", portfolio_size=1, n_groups=2, n_random=30)
        self.assertEqual(set(simulations["Spread bps"].round(6)), {-300.0, 300.0})
        self.assertTrue(simulations["Sharpe"].isna().all())
