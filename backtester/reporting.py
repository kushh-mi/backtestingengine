"""Save an auditable bundle of backtest outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .analytics import monthly_returns, yearly_returns
from .engine import BacktestResult


def save_result(
    result: BacktestResult,
    output_directory: Path,
    configuration: Mapping[str, object],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(output_directory / "equity_curve.csv")
    result.orders.to_csv(output_directory / "order_lifecycle.csv", index=False)
    result.fills.to_csv(output_directory / "fills.csv", index=False)
    result.signals.to_csv(output_directory / "signals.csv")
    result.final_positions.to_csv(output_directory / "final_positions.csv")
    pd.Series(result.metrics, name="value").to_csv(output_directory / "metrics.csv")
    monthly_returns(result.equity_curve["equity"]).to_csv(
        output_directory / "monthly_returns.csv"
    )
    yearly_returns(result.equity_curve["equity"]).to_csv(
        output_directory / "yearly_returns.csv"
    )
    with (output_directory / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(configuration), handle, indent=2, sort_keys=True)


def save_comparison_chart(
    strategy: BacktestResult,
    benchmark: BacktestResult,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    strategy_growth = strategy.equity_curve["equity"] / strategy.equity_curve["equity"].iloc[0]
    benchmark_growth = benchmark.equity_curve["equity"] / benchmark.equity_curve["equity"].iloc[0]
    figure, axis = plt.subplots(figsize=(11, 6))
    strategy_growth.plot(ax=axis, label="Momentum strategy", linewidth=2)
    benchmark_growth.plot(ax=axis, label="Buy and hold", linewidth=1.5)
    axis.set(title="Growth of £1", xlabel="Date", ylabel="Portfolio value (£)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

