"""Cash, positions, corporate actions and portfolio accounting."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .events import FillEvent, MarketEvent, OrderSide, PortfolioSnapshot, Position


class Portfolio:
    def __init__(self, starting_cash: float, tickers: tuple[str, ...]) -> None:
        if starting_cash <= 0:
            raise ValueError("Starting cash must be positive.")
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.positions = {ticker: Position(ticker) for ticker in tickers}
        self.last_prices: dict[str, float] = {}
        self.total_commission = 0.0
        self.total_spread_cost = 0.0
        self.total_slippage_cost = 0.0
        self.dividend_income = 0.0
        self.total_interest = 0.0
        self.snapshots: list[PortfolioSnapshot] = []
        self.fill_rows: list[dict[str, object]] = []

    def accrue_interest(
        self,
        annual_cash_rate: float,
        annual_borrow_rate: float,
        periods_per_year: int = 252,
    ) -> float:
        """Credit positive cash or charge negative cash once per market session."""

        annual_rate = annual_cash_rate if self.cash >= 0 else annual_borrow_rate
        if annual_rate <= -1:
            raise ValueError("An annual interest rate cannot be -100% or lower.")
        daily_rate = (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0
        interest = self.cash * daily_rate
        self.cash += interest
        self.total_interest += interest
        return interest

    def apply_corporate_actions(self, event: MarketEvent) -> None:
        """Apply splits first and then pay dividends to current shareholders."""

        for ticker, bar in event.bars.items():
            position = self.positions[ticker]
            if bar.split_ratio != 1.0 and position.quantity:
                position.quantity *= bar.split_ratio
                position.average_cost /= bar.split_ratio
            if bar.dividend and position.quantity:
                payment = position.quantity * bar.dividend
                self.cash += payment
                self.dividend_income += payment

    def can_apply_fill(self, side: OrderSide, quantity: float, price: float, commission: float, ticker: str) -> bool:
        if quantity <= 0:
            return False
        if side == OrderSide.BUY:
            return self.cash + 1e-9 >= quantity * price + commission
        return self.positions[ticker].quantity + 1e-9 >= quantity

    def apply_fill(self, fill: FillEvent) -> None:
        position = self.positions[fill.ticker]
        quantity = fill.quantity
        if not self.can_apply_fill(fill.side, quantity, fill.fill_price, fill.commission, fill.ticker):
            raise ValueError("Fill violates cash or holdings constraints.")

        if fill.side == OrderSide.BUY:
            old_cost = position.quantity * position.average_cost
            new_cost = quantity * fill.fill_price
            position.quantity += quantity
            position.average_cost = (old_cost + new_cost) / position.quantity
            self.cash -= new_cost + fill.commission
            realised = 0.0
        else:
            proceeds = quantity * fill.fill_price
            realised = quantity * (fill.fill_price - position.average_cost) - fill.commission
            position.realised_pnl += realised
            position.quantity -= quantity
            self.cash += proceeds - fill.commission
            if abs(position.quantity) < 1e-10:
                position.quantity = 0.0
                position.average_cost = 0.0

        self.total_commission += fill.commission
        self.total_spread_cost += fill.spread_cost
        self.total_slippage_cost += fill.slippage_cost
        self.fill_rows.append(
            {
                "timestamp": fill.timestamp,
                "order_id": fill.order_id,
                "ticker": fill.ticker,
                "side": fill.side.value,
                "quantity": quantity,
                "reference_price": fill.reference_price,
                "fill_price": fill.fill_price,
                "notional": fill.notional,
                "commission": fill.commission,
                "spread_cost": fill.spread_cost,
                "slippage_cost": fill.slippage_cost,
                "realised_pnl": realised,
                "cash_after": self.cash,
            }
        )

    def mark_to_market(self, event: MarketEvent) -> PortfolioSnapshot:
        for ticker, bar in event.bars.items():
            self.last_prices[ticker] = bar.close
        position_values = {
            ticker: position.quantity * self.last_prices.get(ticker, 0.0)
            for ticker, position in self.positions.items()
        }
        market_value = sum(position_values.values())
        gross_exposure = sum(abs(value) for value in position_values.values())
        net_exposure = market_value
        equity = self.cash + market_value
        if equity < -1e-8:
            raise RuntimeError("Portfolio equity became negative.")
        snapshot = PortfolioSnapshot(
            timestamp=event.timestamp,
            cash=self.cash,
            market_value=market_value,
            equity=equity,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            total_commission=self.total_commission,
            total_spread_cost=self.total_spread_cost,
            total_slippage_cost=self.total_slippage_cost,
            total_interest=self.total_interest,
            position_values=position_values,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def equity(self, prices: dict[str, float] | None = None) -> float:
        marks = self.last_prices if prices is None else prices
        return self.cash + sum(
            position.quantity * marks.get(ticker, self.last_prices.get(ticker, 0.0))
            for ticker, position in self.positions.items()
        )

    def snapshots_frame(self) -> pd.DataFrame:
        rows = []
        for snapshot in self.snapshots:
            row = asdict(snapshot)
            values = row.pop("position_values")
            row.update({f"value_{ticker}": value for ticker, value in values.items()})
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("timestamp")

    def fills_frame(self) -> pd.DataFrame:
        if not self.fill_rows:
            return pd.DataFrame(
                columns=[
                    "timestamp", "order_id", "ticker", "side", "quantity",
                    "reference_price", "fill_price", "notional", "commission",
                    "spread_cost", "slippage_cost", "realised_pnl", "cash_after",
                ]
            )
        return pd.DataFrame(self.fill_rows)

