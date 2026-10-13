"""Conversion of desired portfolio weights into executable orders."""

from __future__ import annotations

import math

import pandas as pd

from .events import Order, OrderSide, OrderType, TimeInForce
from .portfolio import Portfolio


class TargetWeightSizer:
    def __init__(self, allow_fractional_shares: bool = False) -> None:
        self.allow_fractional_shares = allow_fractional_shares
        self._next_order_id = 1

    def create_orders(
        self,
        timestamp: pd.Timestamp,
        target_weights: dict[str, float],
        close_prices: dict[str, float],
        portfolio: Portfolio,
    ) -> list[Order]:
        equity = portfolio.equity(close_prices)
        if equity <= 0:
            return []
        orders: list[Order] = []
        for ticker, target_weight in target_weights.items():
            if ticker not in close_prices:
                continue
            price = close_prices[ticker]
            exact_quantity = equity * target_weight / price
            desired_quantity = (
                exact_quantity
                if self.allow_fractional_shares
                else float(math.floor(exact_quantity + 1e-12))
            )
            current_quantity = portfolio.positions[ticker].quantity
            difference = desired_quantity - current_quantity
            if abs(difference) < 1e-10:
                continue
            order = Order(
                order_id=self._next_order_id,
                created_at=timestamp,
                ticker=ticker,
                side=OrderSide.BUY if difference > 0 else OrderSide.SELL,
                quantity=abs(difference),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )
            self._next_order_id += 1
            orders.append(order)
        # Selling first frees cash before buys execute at the next session.
        return sorted(orders, key=lambda order: order.side != OrderSide.SELL)

