"""Historical market-data loading and chronological access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from .events import MarketBar, MarketEvent


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
OPTIONAL_DEFAULTS = {"Dividends": 0.0, "Stock Splits": 0.0}


def _normalise_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    data = frame.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.sort_index()
    if data.index.has_duplicates:
        raise ValueError(f"{ticker} contains duplicate dates.")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{ticker} is missing required columns: {missing}")
    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in data.columns:
            data[column] = default

    numeric = list(REQUIRED_COLUMNS) + list(OPTIONAL_DEFAULTS)
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=list(REQUIRED_COLUMNS))
    if data.empty:
        raise ValueError(f"{ticker} has no valid market rows.")
    if (data[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError(f"{ticker} contains a non-positive price.")
    if (data["High"] < data[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError(f"{ticker} has a high below another daily price.")
    if (data["Low"] > data[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError(f"{ticker} has a low above another daily price.")
    if (data["Volume"] < 0).any():
        raise ValueError(f"{ticker} contains negative volume.")
    if (data["Stock Splits"] < 0).any():
        raise ValueError(f"{ticker} contains a negative split ratio.")
    return data


@dataclass
class MarketDataPortal:
    """Validated OHLCV data exposed one chronological session at a time."""

    frames: Mapping[str, pd.DataFrame]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("At least one ticker is required.")
        self.frames = {
            ticker: _normalise_frame(frame, ticker)
            for ticker, frame in self.frames.items()
        }
        all_dates = sorted(set().union(*(frame.index for frame in self.frames.values())))
        self.dates = pd.DatetimeIndex(all_dates)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.frames)

    def events(self) -> Iterator[MarketEvent]:
        for timestamp in self.dates:
            bars: dict[str, MarketBar] = {}
            for ticker, frame in self.frames.items():
                if timestamp not in frame.index:
                    continue
                row = frame.loc[timestamp]
                split = float(row["Stock Splits"])
                bars[ticker] = MarketBar(
                    ticker=ticker,
                    timestamp=timestamp,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    adjusted_close=float(row["Adj Close"]),
                    volume=float(row["Volume"]),
                    dividend=float(row["Dividends"]),
                    split_ratio=split if split > 0 else 1.0,
                )
            if bars:
                yield MarketEvent(timestamp, bars)


def from_close_prices(prices: pd.DataFrame, volume: float = 1_000_000.0) -> MarketDataPortal:
    """Create valid bars from close-only data for tests and simple experiments."""

    frames = {}
    for ticker in prices.columns:
        close = prices[ticker].dropna().astype(float)
        frames[ticker] = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
            index=close.index,
        )
    return MarketDataPortal(frames)


def make_demo_ohlcv(
    tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA", "EEM", "AGG"),
    start: str = "2007-01-01",
    end: str = "2025-12-31",
    seed: int = 7,
) -> MarketDataPortal:
    """Create deterministic, explicitly artificial OHLCV teaching data."""

    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    market_noise = rng.normal(0.0, 0.008, size=len(dates))
    asset_noise = rng.normal(0.0, 0.005, size=(len(dates), len(tickers)))
    returns = 0.00025 + 0.75 * market_noise[:, None] + asset_noise
    crisis = (dates >= "2007-10-01") & (dates <= "2009-06-30")
    shock = (dates >= "2020-02-01") & (dates <= "2020-05-31")
    returns[crisis] += -0.0008 + rng.normal(0.0, 0.009, (crisis.sum(), len(tickers)))
    returns[shock] += -0.0015 + rng.normal(0.0, 0.014, (shock.sum(), len(tickers)))

    closes = 100.0 * pd.DataFrame(1.0 + returns, index=dates, columns=tickers).cumprod()
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        close = closes[ticker]
        overnight = rng.normal(0.0, 0.0015, len(dates))
        open_price = close.shift(1).fillna(close.iloc[0]) * (1.0 + overnight)
        intraday_width = np.abs(rng.normal(0.003, 0.0015, len(dates)))
        high = pd.concat([open_price, close], axis=1).max(axis=1) * (1 + intraday_width)
        low = pd.concat([open_price, close], axis=1).min(axis=1) * (1 - intraday_width)
        frames[ticker] = pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": rng.integers(500_000, 5_000_000, len(dates)),
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
            index=dates,
        )
    return MarketDataPortal(frames)


def download_yfinance_data(
    tickers: tuple[str, ...], start: str, end: str
) -> MarketDataPortal:
    """Download unadjusted execution prices plus adjusted closes and actions."""

    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError(
            "Real-data mode needs yfinance. Run: pip install -r requirements.txt"
        ) from error

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
        )
        if frame.empty:
            raise RuntimeError(f"No data downloaded for {ticker}.")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        # Some yfinance download versions omit actions; defaults are explicit.
        frames[ticker] = frame
    return MarketDataPortal(frames)

