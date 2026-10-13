"""Small correctness checks using Python's built-in unittest library."""

import unittest

import numpy as np
import pandas as pd

from momentum_project import (
    backtest,
    moving_average_signal,
    parameter_stress_test,
    performance_statistics,
)


class MomentumProjectTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        self.rising_prices = pd.DataFrame(
            {"ETF_A": np.arange(100.0, 400.0)}, index=dates
        )

    def test_signal_waits_for_the_long_window(self):
        signal = moving_average_signal(self.rising_prices, 5, 20)
        self.assertTrue((signal.iloc[:19] == 0.0).all().all())
        self.assertEqual(signal.iloc[19, 0], 1.0)

    def test_signal_is_shifted_before_it_earns_a_return(self):
        result = backtest(self.rising_prices, 5, 20, transaction_cost_bps=0)
        # The signal first turns on at row 19; the position starts at row 20.
        self.assertEqual(result["weights"].iloc[19, 0], 0.0)
        self.assertEqual(result["weights"].iloc[20, 0], 1.0)

    def test_maximum_drawdown_for_known_path(self):
        returns = pd.Series([0.10, -0.20, 0.05])
        stats = performance_statistics(returns)
        self.assertAlmostEqual(stats["maximum_drawdown"], 0.20)

    def test_stress_test_counts_every_combination(self):
        table = parameter_stress_test(
            self.rising_prices,
            short_windows=(5, 10),
            long_windows=(20, 30, 40),
            transaction_cost_bps=0,
        )
        self.assertEqual(len(table), 6)


if __name__ == "__main__":
    unittest.main()

