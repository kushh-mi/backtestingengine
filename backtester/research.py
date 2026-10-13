"""Regime, parameter-grid and walk-forward research using the full engine."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .analytics import calculate_metrics
from .data import MarketDataPortal
from .engine import BacktestEngine, BacktestResult
from .strategy import MovingAverageMomentumStrategy


def regime_analysis(
    result: BacktestResult,
    regimes: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for name, (start, end) in regimes.items():
        equity = result.equity_curve.loc[start:end, "equity"]
        if len(equity) < 2:
            continue
        metrics = calculate_metrics(equity)
        rows.append({"regime": name, "start": equity.index[0], "end": equity.index[-1], **metrics})
    return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


def parameter_grid(
    data: MarketDataPortal,
    tickers: tuple[str, ...],
    short_windows: tuple[int, ...],
    long_windows: tuple[int, ...],
    engine_factory: Callable[[object], BacktestEngine],
) -> pd.DataFrame:
    """Run each valid pair through an independent full event-driven simulation."""

    rows = []
    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue
            result = engine_factory(MovingAverageMomentumStrategy(tickers, short, long)).run()
            rows.append({"short_window": short, "long_window": long, **result.metrics})
    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

