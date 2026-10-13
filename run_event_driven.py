"""Run the full event-driven ETF backtester from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtester.broker import SimulatedBroker
from backtester.costs import CostModel
from backtester.data import download_yfinance_data, make_demo_ohlcv
from backtester.engine import BacktestEngine
from backtester.reporting import save_comparison_chart, save_result
from backtester.risk import RiskManager
from backtester.strategy import BuyAndHoldStrategy, MovingAverageMomentumStrategy


TICKERS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "AGG")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="use artificial teaching data")
    source.add_argument("--real-data", action="store_true", help="download historical Yahoo data")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--short", type=int, default=50)
    parser.add_argument("--long", type=int, default=200)
    parser.add_argument("--starting-cash", type=float, default=100_000.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--spread-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--max-volume-participation", type=float, default=0.10)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--cash-rate", type=float, default=0.0)
    parser.add_argument("--borrow-rate", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_event_driven"))
    return parser.parse_args()


def make_engine(data, strategy, args: argparse.Namespace) -> BacktestEngine:
    cost_model = CostModel(
        commission_bps=args.commission_bps,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )
    broker = SimulatedBroker(cost_model, args.max_volume_participation)
    risk = RiskManager(
        max_position_weight=1.0 / len(TICKERS),
        max_gross_exposure=1.0,
        min_cash_weight=0.0,
        long_only=True,
    )
    return BacktestEngine(
        data=data,
        strategy=strategy,
        starting_cash=args.starting_cash,
        broker=broker,
        risk_manager=risk,
        annual_risk_free_rate=args.risk_free_rate,
        annual_cash_rate=args.cash_rate,
        annual_borrow_rate=args.borrow_rate,
    )


def main() -> None:
    args = arguments()
    data = (
        make_demo_ohlcv(TICKERS, args.start, args.end)
        if args.demo
        else download_yfinance_data(TICKERS, args.start, args.end)
    )
    strategy = make_engine(
        data,
        MovingAverageMomentumStrategy(TICKERS, args.short, args.long),
        args,
    ).run()
    benchmark = make_engine(data, BuyAndHoldStrategy(TICKERS), args).run()

    config = {
        "data": "artificial_demo" if args.demo else "yahoo_finance",
        "tickers": list(TICKERS),
        "start": args.start,
        "end": args.end,
        "short_window": args.short,
        "long_window": args.long,
        "starting_cash": args.starting_cash,
        "commission_bps": args.commission_bps,
        "spread_bps": args.spread_bps,
        "slippage_bps": args.slippage_bps,
        "max_volume_participation": args.max_volume_participation,
        "annual_risk_free_rate": args.risk_free_rate,
        "annual_cash_rate": args.cash_rate,
        "annual_borrow_rate": args.borrow_rate,
        "execution_timing": "signal at close, earliest fill at next session open",
        "fractional_shares": False,
    }
    save_result(strategy, args.output_dir / "strategy", config)
    save_result(benchmark, args.output_dir / "benchmark", config)
    save_comparison_chart(strategy, benchmark, args.output_dir / "comparison.png")

    summary = pd.DataFrame(
        {"strategy": strategy.metrics, "benchmark": benchmark.metrics}
    )
    summary["difference"] = summary["strategy"] - summary["benchmark"]
    summary.to_csv(args.output_dir / "comparison_metrics.csv")

    def percentage(value: float) -> str:
        return f"{100 * value:.2f}%"

    print("\nEVENT-DRIVEN BACKTEST")
    print(f"Strategy annualised return: {percentage(strategy.metrics['annualised_return'])}")
    print(f"Strategy Sharpe ratio:      {strategy.metrics['sharpe_ratio']:.2f}")
    print(f"Strategy maximum drawdown:  {percentage(strategy.metrics['maximum_drawdown'])}")
    print(f"Benchmark annualised return:{percentage(benchmark.metrics['annualised_return']):>10}")
    print(f"Completed strategy fills:   {int(strategy.metrics['fill_count'])}")
    print(f"Total commissions:          £{strategy.metrics['total_commission']:.2f}")
    print(f"\nAudit files saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

