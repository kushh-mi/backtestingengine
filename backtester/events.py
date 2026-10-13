"""Data objects passed between the engine's components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"


@dataclass(frozen=True)
class MarketBar:
    """One asset's prices and corporate actions for one trading session."""

    ticker: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float
    dividend: float = 0.0
    split_ratio: float = 1.0


@dataclass(frozen=True)
class MarketEvent:
    timestamp: pd.Timestamp
    bars: Mapping[str, MarketBar]


@dataclass(frozen=True)
class SignalEvent:
    """Desired end-state weights produced after observing a market close."""

    timestamp: pd.Timestamp
    target_weights: Mapping[str, float]
    reason: str


@dataclass
class Order:
    """Instruction sent to the simulated broker."""

    order_id: int
    created_at: pd.Timestamp
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: float = 0.0
    rejection_reason: str | None = None

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)


@dataclass(frozen=True)
class FillEvent:
    """The broker's simulated execution of some or all of an order."""

    order_id: int
    timestamp: pd.Timestamp
    ticker: str
    side: OrderSide
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    spread_cost: float
    slippage_cost: float

    @property
    def notional(self) -> float:
        return self.quantity * self.fill_price


@dataclass(frozen=True)
class OrderRecord:
    timestamp: pd.Timestamp
    order_id: int
    ticker: str
    side: str
    quantity: float
    order_type: str
    status: str
    reason: str | None = None


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0
    average_cost: float = 0.0
    realised_pnl: float = 0.0


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    cash: float
    market_value: float
    equity: float
    gross_exposure: float
    net_exposure: float
    total_commission: float
    total_spread_cost: float
    total_slippage_cost: float
    total_interest: float
    position_values: Mapping[str, float] = field(default_factory=dict)

