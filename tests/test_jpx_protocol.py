import unittest

import numpy as np
import pandas as pd

from src.evaluation import evaluate_ranked_books, jpx_sharpe
from src.diagnostics import paired_sharpe_interval
from src.splitting import development_windows, purged_window


class JpxProtocolTest(unittest.TestCase):
    def test_paired_bootstrap_is_invariant_to_units(self):
        spread = np.array([.014, -.025, .033, .007, -.004, .015])
        paired = pd.DataFrame({"Candidate": spread, "Baseline": spread * 200})
        np.testing.assert_allclose(paired_sharpe_interval(paired, block_size=3), [0., 0.], atol=1e-12)

    def test_missing_label_never_replaces_selected_security(self):
        dates = pd.date_range("2020-01-01", periods=3)
        signals = pd.DataFrame({
            "Date": np.repeat(dates, 5), "SecuritiesCode": np.tile([5, 1, 2, 4, 3], 3),
        })
        scores = pd.DataFrame({"A": [1, 1, .5, 0, 0] * 3, "B": [1, 0, 0, 1, .5] * 3})
        labels = signals.assign(Target=[.02, .06, np.nan, -.02, .01] * 3)
        labels.loc[6, "Target"] = np.nan
        ranks, daily = evaluate_ranked_books(signals, scores, labels, portfolio_size=2)
        first = ranks.loc[ranks["Date"].eq(dates[0]) & ranks["Model"].eq("A")]
        self.assertEqual(first.sort_values("Rank")["SecuritiesCode"].tolist(), [1, 5, 2, 3, 4])
        a = daily.loc[daily["Model"].eq("A")].set_index("Date")
        self.assertAlmostEqual(a.loc[dates[0], "Spread"], (2 * .06 + .02 - 2 * -.02 - .01) / 3)
        self.assertTrue(np.isnan(a.loc[dates[1], "Spread"]))
        self.assertEqual(a.loc[dates[1], "Missing selected labels"], 1)
        self.assertAlmostEqual(a.loc[dates[2], "Spread"], a.loc[dates[0], "Spread"])

    def test_reference_scaling_and_undefined_sharpe(self):
        values = pd.Series([.014, -.025, .033, .007])
        reference = values * 200
        self.assertAlmostEqual(jpx_sharpe(values), reference.mean() / reference.std(ddof=1))
        self.assertTrue(np.isnan(jpx_sharpe(pd.Series([0., 0.]))))
        self.assertTrue(np.isnan(jpx_sharpe(pd.Series([.01, np.nan]))))

    def test_incomplete_book_is_reported_and_missing_scores_rejected(self):
        keys = pd.DataFrame({"Date": pd.Timestamp("2020-01-01"), "SecuritiesCode": [1, 2, 3]})
        labels = keys.assign(Target=[.01, -.02, .04])
        _, daily = evaluate_ranked_books(keys, pd.DataFrame({"A": [1., 0., -1.]}), labels, portfolio_size=2)
        self.assertFalse(daily.loc[0, "Active"])
        self.assertTrue(pd.isna(daily.loc[0, "Spread"]))
        with self.assertRaisesRegex(ValueError, "entire eligible universe"):
            evaluate_ranked_books(keys, pd.DataFrame({"A": [1., 0., np.nan]}), labels, portfolio_size=2)

    def test_development_windows_are_contiguous_and_purge_by_label_end(self):
        windows = development_windows("2021-12-09")
        self.assertEqual(len(windows), 6)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(windows, windows[1:])))
        dates = pd.bdate_range("2021-06-24", periods=12)
        labels = pd.DataFrame({"Date": dates[:-2], "LabelEndDate": dates[2:]})
        fit, evaluation = purged_window(labels, "2021-07-01", "2021-07-08")
        self.assertTrue(fit.LabelEndDate.lt(pd.Timestamp("2021-07-01")).all())
        self.assertTrue(evaluation.LabelEndDate.lt(pd.Timestamp("2021-07-08")).all())
        self.assertEqual(evaluation.Date.iloc[0], pd.Timestamp("2021-07-01"))
