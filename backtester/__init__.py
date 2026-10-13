"""Reusable event-driven backtesting package."""

from .analytics import calculate_metrics
from .broker import SimulatedBroker
from .data import MarketDataPortal, make_demo_ohlcv
from .engine import BacktestEngine, BacktestResult
from .events import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from .portfolio import Portfolio
from .risk import RiskManager
from .strategy import (
    BuyAndHoldStrategy,
    MovingAverageMomentumStrategy,
    Strategy,
    WalkForwardMomentumStrategy,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BuyAndHoldStrategy",
    "MarketDataPortal",
    "MovingAverageMomentumStrategy",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Portfolio",
    "RiskManager",
    "SimulatedBroker",
    "Strategy",
    "TimeInForce",
    "WalkForwardMomentumStrategy",
    "calculate_metrics",
    "make_demo_ohlcv",
]

