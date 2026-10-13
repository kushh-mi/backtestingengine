"""Order lifecycle and execution simulation."""

from __future__ import annotations

import math

from .costs import CostModel
from .events import (
    FillEvent,
    MarketBar,
    MarketEvent,
    Order,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from .portfolio import Portfolio


class SimulatedBroker:
    def __init__(
        self,
        cost_model: CostModel | None = None,
        max_volume_participation: float = 0.10,
    ) -> None:
        if not 0 < max_volume_participation <= 1:
            raise ValueError("Volume participation must be in (0, 1].")
        self.cost_model = cost_model or CostModel()
        self.max_volume_participation = max_volume_participation
        self.open_orders: list[Order] = []
        self.all_orders: list[Order] = []
        self.records: list[OrderRecord] = []

    def _record(self, order: Order, timestamp, reason: str | None = None) -> None:
        self.records.append(
            OrderRecord(
                timestamp=timestamp,
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
                status=order.status.value,
                reason=reason,
            )
        )

    def submit(self, order: Order) -> None:
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Quantity must be positive."
            self.all_orders.append(order)
            self._record(order, order.created_at, order.rejection_reason)
            return
        if order.order_type == OrderType.LIMIT and (
            order.limit_price is None or order.limit_price <= 0
        ):
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "A limit order needs a positive limit price."
            self.all_orders.append(order)
            self._record(order, order.created_at, order.rejection_reason)
            return
        if order.order_type == OrderType.STOP and (
            order.stop_price is None or order.stop_price <= 0
        ):
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "A stop order needs a positive stop price."
            self.all_orders.append(order)
            self._record(order, order.created_at, order.rejection_reason)
            return
        order.status = OrderStatus.ACCEPTED
        self.open_orders.append(order)
        self.all_orders.append(order)
        self._record(order, order.created_at)

    def cancel_open_orders(self, timestamp, reason: str = "Cancelled") -> None:
        for order in self.open_orders:
            order.status = OrderStatus.CANCELLED
            self._record(order, timestamp, reason)
        self.open_orders.clear()

    def adjust_orders_for_splits(self, event: MarketEvent) -> None:
        for order in self.open_orders:
            bar = event.bars.get(order.ticker)
            if bar is None or bar.split_ratio == 1.0:
                continue
            order.quantity *= bar.split_ratio
            order.filled_quantity *= bar.split_ratio
            if order.limit_price is not None:
                order.limit_price /= bar.split_ratio
            if order.stop_price is not None:
                order.stop_price /= bar.split_ratio

    @staticmethod
    def _reference_price(order: Order, bar: MarketBar) -> float | None:
        if order.order_type == OrderType.MARKET:
            return bar.open
        if order.order_type == OrderType.LIMIT:
            assert order.limit_price is not None
            if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                return min(bar.open, order.limit_price)
            if order.side == OrderSide.SELL and bar.high >= order.limit_price:
                return max(bar.open, order.limit_price)
            return None
        assert order.stop_price is not None
        if order.side == OrderSide.BUY and bar.high >= order.stop_price:
            return max(bar.open, order.stop_price)
        if order.side == OrderSide.SELL and bar.low <= order.stop_price:
            return min(bar.open, order.stop_price)
        return None

    def _affordable_buy_quantity(self, desired: float, fill_price: float, portfolio: Portfolio) -> float:
        available_after_fixed = portfolio.cash - self.cost_model.commission_fixed
        if available_after_fixed <= 0:
            return 0.0
        variable_rate = self.cost_model.commission_bps / 10_000.0
        maximum = available_after_fixed / (fill_price * (1.0 + variable_rate))
        return min(desired, float(math.floor(maximum + 1e-12)))

    def process_market_event(self, event: MarketEvent, portfolio: Portfolio) -> list[FillEvent]:
        """Attempt eligible orders at this session's prices; sells run before buys."""

        fills: list[FillEvent] = []
        still_open: list[Order] = []
        orders = sorted(self.open_orders, key=lambda order: order.side != OrderSide.SELL)
        for order in orders:
            # Orders created after a close are never eligible in that same session.
            if event.timestamp <= order.created_at:
                still_open.append(order)
                continue
            bar = event.bars.get(order.ticker)
            if bar is None:
                if order.time_in_force == TimeInForce.DAY:
                    order.status = OrderStatus.CANCELLED
                    self._record(order, event.timestamp, "No market bar before DAY expiry.")
                else:
                    still_open.append(order)
                continue
            reference = self._reference_price(order, bar)
            if reference is None:
                if order.time_in_force == TimeInForce.DAY:
                    order.status = OrderStatus.CANCELLED
                    self._record(order, event.timestamp, "Order condition not reached.")
                else:
                    still_open.append(order)
                continue

            theoretical_price, spread_per_unit, slippage_per_unit = self.cost_model.execution_price(
                reference, order.side
            )
            if order.order_type == OrderType.LIMIT:
                assert order.limit_price is not None
                fill_price = (
                    min(theoretical_price, order.limit_price)
                    if order.side == OrderSide.BUY
                    else max(theoretical_price, order.limit_price)
                )
            else:
                fill_price = theoretical_price

            volume_limit = bar.volume * self.max_volume_participation
            quantity = min(order.remaining_quantity, volume_limit)
            if order.side == OrderSide.BUY:
                quantity = self._affordable_buy_quantity(quantity, fill_price, portfolio)
            else:
                quantity = min(quantity, portfolio.positions[order.ticker].quantity)
            if quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Insufficient cash, holdings, or market volume."
                self._record(order, event.timestamp, order.rejection_reason)
                continue

            commission = self.cost_model.commission(quantity * fill_price)
            fill = FillEvent(
                order_id=order.order_id,
                timestamp=event.timestamp,
                ticker=order.ticker,
                side=order.side,
                quantity=quantity,
                reference_price=reference,
                fill_price=fill_price,
                commission=commission,
                spread_cost=quantity * spread_per_unit,
                slippage_cost=quantity * slippage_per_unit,
            )
            portfolio.apply_fill(fill)
            fills.append(fill)
            order.filled_quantity += quantity
            if order.remaining_quantity <= 1e-10:
                order.status = OrderStatus.FILLED
                self._record(order, event.timestamp)
            elif order.time_in_force == TimeInForce.DAY:
                order.status = OrderStatus.CANCELLED
                self._record(order, event.timestamp, "Unfilled DAY remainder cancelled.")
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                self._record(order, event.timestamp)
                still_open.append(order)
        self.open_orders = still_open
        return fills

    def orders_frame(self):
        import pandas as pd

        return pd.DataFrame([record.__dict__ for record in self.records])

