"""Run regimes, 64 full-engine parameter tests, and walk-forward validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtester.broker import SimulatedBroker
from backtester.costs import CostModel
from backtester.data import download_yfinance_data, make_demo_ohlcv
from backtester.engine import BacktestEngine
from backtester.reporting import save_result
from backtester.research import parameter_grid, regime_analysis
from backtester.risk import RiskManager
from backtester.strategy import WalkForwardMomentumStrategy


TICKERS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "AGG")
SHORT_WINDOWS = (5, 10, 15, 20, 30, 40, 50, 60)
LONG_WINDOWS = (80, 100, 120, 150, 180, 200, 220, 250)
REGIMES = {
    "Global financial crisis": ("2007-10-01", "2009-06-30"),
    "Long expansion": ("2009-07-01", "2019-12-31"),
    "Pandemic and inflation shock": ("2020-01-01", "2022-12-31"),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--real-data", action="store_true")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--starting-cash", type=float, default=100_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_event_research"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data = (
        make_demo_ohlcv(TICKERS, args.start, args.end)
        if args.demo
        else download_yfinance_data(TICKERS, args.start, args.end)
    )

    def engine_factory(strategy) -> BacktestEngine:
        return BacktestEngine(
            data,
            strategy,
            starting_cash=args.starting_cash,
            broker=SimulatedBroker(CostModel()),
            risk_manager=RiskManager(max_position_weight=1 / len(TICKERS)),
        )

    print("Running 64 independent event-driven parameter tests...")
    grid = parameter_grid(data, TICKERS, SHORT_WINDOWS, LONG_WINDOWS, engine_factory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "parameter_grid.csv", index=False)

    walk_strategy = WalkForwardMomentumStrategy(
        TICKERS,
        SHORT_WINDOWS,
        LONG_WINDOWS,
        training_days=5 * 252,
        reselection_days=252,
    )
    walk_result = engine_factory(walk_strategy).run()
    configuration = {
        "mode": "walk_forward",
        "tickers": list(TICKERS),
        "training_days": 5 * 252,
        "reselection_days": 252,
        "short_windows": list(SHORT_WINDOWS),
        "long_windows": list(LONG_WINDOWS),
        "data": "artificial_demo" if args.demo else "yahoo_finance",
    }
    save_result(walk_result, args.output_dir / "walk_forward", configuration)
    pd.DataFrame(walk_strategy.selection_log).to_csv(
        args.output_dir / "walk_forward" / "parameter_selections.csv", index=False
    )
    regime_analysis(walk_result, REGIMES).to_csv(args.output_dir / "walk_forward_regimes.csv")
    print(f"Completed {len(grid)} parameter combinations.")
    print(f"Walk-forward Sharpe ratio: {walk_result.metrics['sharpe_ratio']:.2f}")
    print(f"Results saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

