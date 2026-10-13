"""Chronological coordinator for data, strategy, risk, orders and accounting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .analytics import calculate_metrics
from .broker import SimulatedBroker
from .data import MarketDataPortal
from .portfolio import Portfolio
from .risk import RiskManager
from .sizing import TargetWeightSizer
from .strategy import Strategy


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict[str, float]
    final_positions: pd.DataFrame


class BacktestEngine:
    def __init__(
        self,
        data: MarketDataPortal,
        strategy: Strategy,
        starting_cash: float = 100_000.0,
        broker: SimulatedBroker | None = None,
        risk_manager: RiskManager | None = None,
        sizer: TargetWeightSizer | None = None,
        annual_risk_free_rate: float = 0.0,
        annual_cash_rate: float = 0.0,
        annual_borrow_rate: float = 0.0,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = Portfolio(starting_cash, data.tickers)
        self.broker = broker or SimulatedBroker()
        self.risk_manager = risk_manager or RiskManager()
        self.sizer = sizer or TargetWeightSizer()
        self.annual_risk_free_rate = annual_risk_free_rate
        self.annual_cash_rate = annual_cash_rate
        self.annual_borrow_rate = annual_borrow_rate
        self.signal_rows: list[dict[str, object]] = []

    def run(self) -> BacktestResult:
        adjusted_history = pd.DataFrame(columns=self.data.tickers, dtype=float)
        last_timestamp = None

        for event in self.data.events():
            last_timestamp = event.timestamp
            if self.portfolio.snapshots:
                self.portfolio.accrue_interest(
                    self.annual_cash_rate, self.annual_borrow_rate
                )
            # Existing shareholders receive actions before this session's orders fill.
            self.portfolio.apply_corporate_actions(event)
            self.broker.adjust_orders_for_splits(event)
            self.broker.process_market_event(event, self.portfolio)
            self.portfolio.mark_to_market(event)

            row = {ticker: event.bars[ticker].adjusted_close for ticker in event.bars}
            adjusted_history.loc[event.timestamp, list(row)] = list(row.values())
            signal = self.strategy.on_market_close(event, adjusted_history)
            if signal is None:
                continue

            targets = self.risk_manager.validate_targets(
                signal.target_weights, self.data.tickers
            )
            # A fresh portfolio decision supersedes any unfilled previous target.
            self.broker.cancel_open_orders(event.timestamp, "Superseded by new target.")
            close_prices = {ticker: bar.close for ticker, bar in event.bars.items()}
            orders = self.sizer.create_orders(
                event.timestamp, targets, close_prices, self.portfolio
            )
            for order in orders:
                self.broker.submit(order)
            signal_row: dict[str, object] = {
                "timestamp": signal.timestamp,
                "reason": signal.reason,
            }
            signal_row.update({f"target_{ticker}": targets[ticker] for ticker in self.data.tickers})
            self.signal_rows.append(signal_row)

        if last_timestamp is None:
            raise ValueError("Market data produced no events.")
        self.broker.cancel_open_orders(last_timestamp, "Backtest ended.")
        equity_curve = self.portfolio.snapshots_frame()
        fills = self.portfolio.fills_frame()
        metrics = calculate_metrics(
            equity_curve["equity"], fills, annual_risk_free_rate=self.annual_risk_free_rate
        )
        metrics["dividend_income"] = float(self.portfolio.dividend_income)
        metrics["cash_interest"] = float(self.portfolio.total_interest)
        metrics["average_gross_exposure"] = float(
            (equity_curve["gross_exposure"] / equity_curve["equity"]).mean()
        )
        final_positions = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "quantity": position.quantity,
                    "average_cost": position.average_cost,
                    "last_price": self.portfolio.last_prices.get(ticker, float("nan")),
                    "market_value": position.quantity * self.portfolio.last_prices.get(ticker, 0.0),
                    "realised_pnl": position.realised_pnl,
                }
                for ticker, position in self.portfolio.positions.items()
            ]
        ).set_index("ticker")
        signals = pd.DataFrame(self.signal_rows)
        if not signals.empty:
            signals = signals.set_index("timestamp")
        return BacktestResult(
            equity_curve=equity_curve,
            orders=self.broker.orders_frame(),
            fills=fills,
            signals=signals,
            metrics=metrics,
            final_positions=final_positions,
        )

