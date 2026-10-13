"""Strategy interface and two example strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .events import MarketEvent, SignalEvent


class Strategy(ABC):
    @abstractmethod
    def on_market_close(
        self, event: MarketEvent, adjusted_close_history: pd.DataFrame
    ) -> SignalEvent | None:
        """Return desired weights after seeing this close, or no new decision."""


class MovingAverageMomentumStrategy(Strategy):
    def __init__(self, tickers: tuple[str, ...], short_window: int = 50, long_window: int = 200) -> None:
        if not 1 <= short_window < long_window:
            raise ValueError("Windows must satisfy 1 <= short < long.")
        self.tickers = tickers
        self.short_window = short_window
        self.long_window = long_window
        self._last_state: tuple[bool, ...] | None = None

    def on_market_close(self, event: MarketEvent, adjusted_close_history: pd.DataFrame) -> SignalEvent | None:
        if len(adjusted_close_history) < self.long_window:
            return None
        recent = adjusted_close_history.reindex(columns=self.tickers)
        if recent.iloc[-self.long_window :].isna().any().any():
            return None
        short_average = recent.iloc[-self.short_window :].mean()
        long_average = recent.iloc[-self.long_window :].mean()
        state = tuple(bool(short_average[ticker] > long_average[ticker]) for ticker in self.tickers)
        if state == self._last_state:
            return None
        self._last_state = state
        allocation = 1.0 / len(self.tickers)
        targets = {
            ticker: allocation if active else 0.0
            for ticker, active in zip(self.tickers, state)
        }
        active = [ticker for ticker, enabled in zip(self.tickers, state) if enabled]
        return SignalEvent(
            timestamp=event.timestamp,
            target_weights=targets,
            reason=f"{self.short_window}/{self.long_window} momentum active: {', '.join(active) or 'none'}",
        )


class BuyAndHoldStrategy(Strategy):
    def __init__(self, tickers: tuple[str, ...]) -> None:
        self.tickers = tickers
        self._submitted = False

    def on_market_close(self, event: MarketEvent, adjusted_close_history: pd.DataFrame) -> SignalEvent | None:
        if self._submitted or not all(ticker in event.bars for ticker in self.tickers):
            return None
        self._submitted = True
        weight = 1.0 / len(self.tickers)
        return SignalEvent(
            timestamp=event.timestamp,
            target_weights={ticker: weight for ticker in self.tickers},
            reason="Initial equal-weight buy-and-hold allocation",
        )


class WalkForwardMomentumStrategy(Strategy):
    """Select moving-average parameters only from trailing historical data."""

    def __init__(
        self,
        tickers: tuple[str, ...],
        short_windows: tuple[int, ...],
        long_windows: tuple[int, ...],
        training_days: int = 5 * 252,
        reselection_days: int = 252,
        selection_cost_bps: float = 5.0,
    ) -> None:
        pairs = [
            (short, long)
            for short in short_windows
            for long in long_windows
            if short < long
        ]
        if not pairs:
            raise ValueError("At least one valid short/long pair is required.")
        if training_days <= 1 or reselection_days <= 0:
            raise ValueError("Training and reselection lengths must be positive.")
        self.tickers = tickers
        self.pairs = pairs
        self.training_days = training_days
        self.reselection_days = reselection_days
        self.selection_cost_bps = selection_cost_bps
        self.max_long = max(long for _, long in pairs)
        self.selected_pair: tuple[int, int] | None = None
        self.next_reselection_row = training_days + self.max_long
        self._last_state: tuple[bool, ...] | None = None
        self.selection_log: list[dict[str, object]] = []

    def _training_sharpe(
        self, history: pd.DataFrame, short: int, long: int
    ) -> float:
        short_average = history.rolling(short).mean()
        long_average = history.rolling(long).mean()
        signal = (short_average > long_average).astype(float)
        signal[long_average.isna()] = 0.0
        weights = signal.shift(1).fillna(0.0) / len(self.tickers)
        returns = history.pct_change().fillna(0.0)
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        portfolio_returns = (weights * returns).sum(axis=1) - turnover * (
            self.selection_cost_bps / 10_000.0
        )
        sample = portfolio_returns.iloc[-self.training_days :]
        volatility = sample.std(ddof=1)
        if not pd.notna(volatility) or volatility <= 0:
            return float("-inf")
        return float(sample.mean() / volatility * (252.0**0.5))

    def _select_parameters(self, history: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        scores = {
            pair: self._training_sharpe(history, *pair) for pair in self.pairs
        }
        selected = max(scores, key=scores.get)
        self.selected_pair = selected
        self.selection_log.append(
            {
                "timestamp": timestamp,
                "short_window": selected[0],
                "long_window": selected[1],
                "training_sharpe": scores[selected],
                "training_start": history.index[-self.training_days],
                "training_end": history.index[-1],
            }
        )

    def on_market_close(
        self, event: MarketEvent, adjusted_close_history: pd.DataFrame
    ) -> SignalEvent | None:
        history = adjusted_close_history.reindex(columns=self.tickers)
        row_number = len(history)
        if row_number < self.training_days + self.max_long:
            return None
        reselected = False
        if self.selected_pair is None or row_number >= self.next_reselection_row:
            self._select_parameters(history, event.timestamp)
            self.next_reselection_row = row_number + self.reselection_days
            reselected = True
        assert self.selected_pair is not None
        short, long = self.selected_pair
        if history.iloc[-long:].isna().any().any():
            return None
        short_average = history.iloc[-short:].mean()
        long_average = history.iloc[-long:].mean()
        state = tuple(short_average[ticker] > long_average[ticker] for ticker in self.tickers)
        if state == self._last_state and not reselected:
            return None
        self._last_state = state
        allocation = 1.0 / len(self.tickers)
        targets = {
            ticker: allocation if active else 0.0
            for ticker, active in zip(self.tickers, state)
        }
        return SignalEvent(
            timestamp=event.timestamp,
            target_weights=targets,
            reason=f"Walk-forward selected {short}/{long}",
        )

