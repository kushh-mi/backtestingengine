"""Correctness tests for execution timing, accounting, costs and risk controls."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtester.broker import SimulatedBroker
from backtester.costs import CostModel
from backtester.data import MarketDataPortal, from_close_prices
from backtester.engine import BacktestEngine
from backtester.events import (
    FillEvent,
    MarketBar,
    MarketEvent,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from backtester.portfolio import Portfolio
from backtester.risk import RiskManager
from backtester.research import parameter_grid
from backtester.strategy import (
    BuyAndHoldStrategy,
    MovingAverageMomentumStrategy,
    WalkForwardMomentumStrategy,
)


def bar(date: str, price: float = 100.0, *, high=None, low=None, volume=1_000.0, dividend=0.0, split=1.0):
    timestamp = pd.Timestamp(date)
    return MarketBar(
        ticker="ETF",
        timestamp=timestamp,
        open=price,
        high=price if high is None else high,
        low=price if low is None else low,
        close=price,
        adjusted_close=price,
        volume=volume,
        dividend=dividend,
        split_ratio=split,
    )


class EventDrivenBacktesterTests(unittest.TestCase):
    def test_market_order_fills_next_session_not_signal_session(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        data = from_close_prices(pd.DataFrame({"ETF": [100.0, 100.0, 110.0]}, index=dates))
        engine = BacktestEngine(
            data,
            BuyAndHoldStrategy(("ETF",)),
            starting_cash=1_000.0,
            broker=SimulatedBroker(CostModel(commission_bps=0, spread_bps=0, slippage_bps=0)),
        )
        result = engine.run()
        self.assertEqual(result.fills.iloc[0]["timestamp"], dates[1])
        self.assertEqual(result.fills.iloc[0]["quantity"], 10.0)
        self.assertAlmostEqual(result.equity_curve.iloc[-1]["equity"], 1_100.0)

    def test_portfolio_buy_and_sell_accounting(self):
        portfolio = Portfolio(1_000.0, ("ETF",))
        buy = FillEvent(1, pd.Timestamp("2024-01-02"), "ETF", OrderSide.BUY, 5, 100, 100, 1, 0, 0)
        portfolio.apply_fill(buy)
        self.assertAlmostEqual(portfolio.cash, 499.0)
        self.assertEqual(portfolio.positions["ETF"].quantity, 5)
        sell = FillEvent(2, pd.Timestamp("2024-01-03"), "ETF", OrderSide.SELL, 2, 110, 110, 1, 0, 0)
        portfolio.apply_fill(sell)
        self.assertAlmostEqual(portfolio.cash, 718.0)
        self.assertAlmostEqual(portfolio.positions["ETF"].realised_pnl, 19.0)

    def test_cost_model_changes_fill_price_and_charges_commission(self):
        portfolio = Portfolio(2_000.0, ("ETF",))
        broker = SimulatedBroker(
            CostModel(commission_bps=10, spread_bps=20, slippage_bps=10),
            max_volume_participation=1.0,
        )
        order = Order(1, pd.Timestamp("2024-01-01"), "ETF", OrderSide.BUY, 10)
        broker.submit(order)
        fills = broker.process_market_event(
            MarketEvent(pd.Timestamp("2024-01-02"), {"ETF": bar("2024-01-02")}), portfolio
        )
        self.assertAlmostEqual(fills[0].fill_price, 100.2)
        self.assertAlmostEqual(fills[0].spread_cost, 1.0)
        self.assertAlmostEqual(fills[0].slippage_cost, 1.0)
        self.assertAlmostEqual(fills[0].commission, 1.002)

    def test_limit_order_waits_until_price_is_reached(self):
        portfolio = Portfolio(2_000.0, ("ETF",))
        broker = SimulatedBroker(CostModel(commission_bps=0, spread_bps=0, slippage_bps=0))
        order = Order(
            1,
            pd.Timestamp("2024-01-01"),
            "ETF",
            OrderSide.BUY,
            5,
            order_type=OrderType.LIMIT,
            limit_price=95.0,
        )
        broker.submit(order)
        no_fill = broker.process_market_event(
            MarketEvent(pd.Timestamp("2024-01-02"), {"ETF": bar("2024-01-02", 100, low=96)}), portfolio
        )
        self.assertEqual(no_fill, [])
        fills = broker.process_market_event(
            MarketEvent(pd.Timestamp("2024-01-03"), {"ETF": bar("2024-01-03", 94, high=101, low=93)}), portfolio
        )
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].fill_price, 94.0)

    def test_stop_order_triggers_after_crossing_stop(self):
        portfolio = Portfolio(2_000.0, ("ETF",))
        broker = SimulatedBroker(CostModel(commission_bps=0, spread_bps=0, slippage_bps=0))
        order = Order(
            1,
            pd.Timestamp("2024-01-01"),
            "ETF",
            OrderSide.BUY,
            5,
            order_type=OrderType.STOP,
            stop_price=105.0,
        )
        broker.submit(order)
        broker.process_market_event(
            MarketEvent(pd.Timestamp("2024-01-02"), {"ETF": bar("2024-01-02", 100, high=104)}), portfolio
        )
        fills = broker.process_market_event(
            MarketEvent(pd.Timestamp("2024-01-03"), {"ETF": bar("2024-01-03", 106, high=108)}), portfolio
        )
        self.assertEqual(fills[0].fill_price, 106.0)

    def test_volume_limit_creates_partial_fill(self):
        portfolio = Portfolio(10_000.0, ("ETF",))
        broker = SimulatedBroker(
            CostModel(commission_bps=0, spread_bps=0, slippage_bps=0),
            max_volume_participation=0.10,
        )
        order = Order(1, pd.Timestamp("2024-01-01"), "ETF", OrderSide.BUY, 50)
        broker.submit(order)
        fills = broker.process_market_event(
            MarketEvent(pd.Timestamp("2024-01-02"), {"ETF": bar("2024-01-02", volume=100)}), portfolio
        )
        self.assertEqual(fills[0].quantity, 10)
        self.assertEqual(order.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(order.remaining_quantity, 40)

    def test_day_order_cancels_if_condition_is_not_reached(self):
        portfolio = Portfolio(2_000.0, ("ETF",))
        broker = SimulatedBroker(CostModel())
        order = Order(
            1,
            pd.Timestamp("2024-01-01"),
            "ETF",
            OrderSide.BUY,
            5,
            order_type=OrderType.LIMIT,
            limit_price=90,
            time_in_force=TimeInForce.DAY,
        )
        broker.submit(order)
        broker.process_market_event(
            MarketEvent(pd.Timestamp("2024-01-02"), {"ETF": bar("2024-01-02", 100, low=99)}), portfolio
        )
        self.assertEqual(order.status, OrderStatus.CANCELLED)

    def test_split_and_dividend_update_accounting(self):
        portfolio = Portfolio(2_000.0, ("ETF",))
        portfolio.apply_fill(
            FillEvent(1, pd.Timestamp("2024-01-01"), "ETF", OrderSide.BUY, 10, 100, 100, 0, 0, 0)
        )
        event = MarketEvent(
            pd.Timestamp("2024-01-02"),
            {"ETF": bar("2024-01-02", 50, dividend=1.0, split=2.0)},
        )
        portfolio.apply_corporate_actions(event)
        self.assertEqual(portfolio.positions["ETF"].quantity, 20)
        self.assertEqual(portfolio.positions["ETF"].average_cost, 50)
        self.assertEqual(portfolio.dividend_income, 20)
        self.assertEqual(portfolio.cash, 1_020)

    def test_risk_manager_rejects_excess_position(self):
        risk = RiskManager(max_position_weight=0.40, max_gross_exposure=1.0)
        with self.assertRaises(ValueError):
            risk.validate_targets({"ETF": 0.5}, ("ETF",))

    def test_future_price_change_cannot_change_earlier_fills(self):
        dates = pd.bdate_range("2024-01-01", periods=8)
        first = pd.DataFrame({"ETF": [10, 11, 12, 13, 14, 15, 16, 17]}, index=dates)
        second = first.copy()
        second.loc[dates[-1], "ETF"] = 1_000

        def run(prices):
            return BacktestEngine(
                from_close_prices(prices),
                MovingAverageMomentumStrategy(("ETF",), 2, 3),
                starting_cash=1_000,
                broker=SimulatedBroker(CostModel(commission_bps=0, spread_bps=0, slippage_bps=0)),
            ).run()

        fills_a = run(first).fills
        fills_b = run(second).fills
        cutoff = dates[-1]
        cols = ["timestamp", "ticker", "side", "quantity", "fill_price"]
        pd.testing.assert_frame_equal(
            fills_a.loc[fills_a["timestamp"] < cutoff, cols].reset_index(drop=True),
            fills_b.loc[fills_b["timestamp"] < cutoff, cols].reset_index(drop=True),
        )

    def test_equity_always_reconciles_to_cash_plus_positions(self):
        dates = pd.bdate_range("2024-01-01", periods=20)
        prices = pd.DataFrame(
            {"A": np.linspace(100, 120, 20), "B": np.linspace(100, 80, 20)},
            index=dates,
        )
        result = BacktestEngine(
            from_close_prices(prices),
            BuyAndHoldStrategy(("A", "B")),
            starting_cash=10_000,
            broker=SimulatedBroker(CostModel(commission_bps=0, spread_bps=0, slippage_bps=0)),
        ).run()
        position_columns = [column for column in result.equity_curve if column.startswith("value_")]
        reconstructed = result.equity_curve["cash"] + result.equity_curve[position_columns].sum(axis=1)
        np.testing.assert_allclose(reconstructed, result.equity_curve["equity"])

    def test_invalid_order_is_rejected(self):
        broker = SimulatedBroker()
        order = Order(1, pd.Timestamp("2024-01-01"), "ETF", OrderSide.BUY, 0)
        broker.submit(order)
        self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_cash_interest_is_compounded(self):
        portfolio = Portfolio(1_000.0, ("ETF",))
        daily = portfolio.accrue_interest(0.10, 0.20)
        self.assertGreater(daily, 0)
        self.assertAlmostEqual(portfolio.cash, 1_000.0 * (1.10 ** (1 / 252)))

    def test_parameter_grid_runs_every_valid_pair(self):
        dates = pd.bdate_range("2020-01-01", periods=60)
        data = from_close_prices(
            pd.DataFrame({"ETF": np.linspace(100, 130, len(dates))}, index=dates)
        )

        def factory(strategy):
            return BacktestEngine(
                data,
                strategy,
                starting_cash=1_000,
                broker=SimulatedBroker(
                    CostModel(commission_bps=0, spread_bps=0, slippage_bps=0)
                ),
            )

        results = parameter_grid(data, ("ETF",), (2, 3), (5, 8), factory)
        self.assertEqual(len(results), 4)

    def test_walk_forward_selection_begins_only_after_training_and_warmup(self):
        dates = pd.bdate_range("2020-01-01", periods=40)
        prices = pd.DataFrame({"ETF": np.linspace(100, 140, 40)}, index=dates)
        strategy = WalkForwardMomentumStrategy(
            ("ETF",), (2, 3), (5, 8), training_days=20, reselection_days=5
        )
        result = BacktestEngine(
            from_close_prices(prices),
            strategy,
            starting_cash=1_000,
            broker=SimulatedBroker(
                CostModel(commission_bps=0, spread_bps=0, slippage_bps=0)
            ),
        ).run()
        self.assertGreaterEqual(strategy.selection_log[0]["timestamp"], dates[27])
        self.assertTrue((result.fills["timestamp"] > strategy.selection_log[0]["timestamp"]).all())


if __name__ == "__main__":
    unittest.main()

