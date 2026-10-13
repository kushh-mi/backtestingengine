"""Educational moving-average momentum research project.

This one file contains the complete research pipeline:

1. obtain adjusted ETF prices (or make reproducible demonstration data),
2. create moving-average buy/cash signals,
3. backtest them without looking into the future,
4. calculate performance and risk statistics,
5. compare the strategy with buy-and-hold,
6. inspect three market regimes,
7. test more than 50 parameter combinations, and
8. perform walk-forward out-of-sample validation.

Run ``python momentum_project.py --demo`` first.  It needs only NumPy and
pandas.  Run ``python momentum_project.py --real-data`` after installing all
packages in requirements.txt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """All important assumptions, kept together so they are easy to change."""

    tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA", "EEM", "AGG")
    start: str = "2007-01-01"
    end: str = "2025-12-31"
    short_window: int = 50
    long_window: int = 200
    transaction_cost_bps: float = 5.0
    trading_days_per_year: int = 252


REGIMES = {
    "Global financial crisis": ("2007-10-01", "2009-06-30"),
    "Long expansion": ("2009-07-01", "2019-12-31"),
    "Pandemic and inflation shock": ("2020-01-01", "2022-12-31"),
}


# Eight short windows times eight long windows = 64 combinations (> 50).
SHORT_WINDOWS = (5, 10, 15, 20, 30, 40, 50, 60)
LONG_WINDOWS = (80, 100, 120, 150, 180, 200, 220, 250)


# ---------------------------------------------------------------------------
# 2. DATA
# ---------------------------------------------------------------------------


def download_adjusted_prices(settings: Settings) -> pd.DataFrame:
    """Download split/dividend-adjusted closing prices and validate the result."""

    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError(
            "Real-data mode needs yfinance. Run: pip install -r requirements.txt"
        ) from error

    raw = yf.download(
        list(settings.tickers),
        start=settings.start,
        end=settings.end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("No prices were downloaded. Check the internet connection.")

    # yfinance normally returns columns such as (Price field, Ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # This branch also makes the function work for a one-ticker experiment.
        prices = raw[["Close"]].rename(columns={"Close": settings.tickers[0]})

    prices = prices.reindex(columns=list(settings.tickers)).dropna(how="all")
    return validate_prices(prices)


def make_demo_prices(settings: Settings, seed: int = 7) -> pd.DataFrame:
    """Create fake but market-like prices so every part can run without internet.

    These are NOT evidence that the strategy would make money.  They are merely
    reproducible teaching data with rising, falling and turbulent periods.
    """

    dates = pd.bdate_range(settings.start, settings.end)
    rng = np.random.default_rng(seed)
    number_of_assets = len(settings.tickers)

    # A shared market movement makes the six assets somewhat correlated.
    market_noise = rng.normal(0.0, 0.008, size=len(dates))
    asset_noise = rng.normal(0.0, 0.005, size=(len(dates), number_of_assets))
    daily_returns = 0.00025 + 0.75 * market_noise[:, None] + asset_noise

    # Insert two deliberately difficult periods to make the demonstration useful.
    crisis = (dates >= "2007-10-01") & (dates <= "2009-06-30")
    shock = (dates >= "2020-02-01") & (dates <= "2020-05-31")
    daily_returns[crisis] += -0.0008 + rng.normal(
        0.0, 0.009, size=(crisis.sum(), number_of_assets)
    )
    daily_returns[shock] += -0.0015 + rng.normal(
        0.0, 0.014, size=(shock.sum(), number_of_assets)
    )

    # If r is a daily return, multiplying by (1+r) updates yesterday's price.
    prices = 100.0 * pd.DataFrame(
        1.0 + daily_returns, index=dates, columns=settings.tickers
    ).cumprod()
    return validate_prices(prices)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Reject data errors that could silently make the research misleading."""

    clean = prices.copy().sort_index().astype(float).ffill().dropna()
    if clean.empty:
        raise ValueError("The price table contains no complete rows.")
    if clean.index.has_duplicates:
        raise ValueError("The price table contains duplicate dates.")
    if (clean <= 0).any().any():
        raise ValueError("Every price must be greater than zero.")
    return clean


# ---------------------------------------------------------------------------
# 3. SIGNALS AND BACKTEST
# ---------------------------------------------------------------------------


def moving_average_signal(
    prices: pd.DataFrame, short_window: int, long_window: int
) -> pd.DataFrame:
    """Return 1 when the short average exceeds the long average, otherwise 0."""

    if not 1 <= short_window < long_window:
        raise ValueError("Windows must satisfy 1 <= short_window < long_window.")

    short_average = prices.rolling(short_window).mean()
    long_average = prices.rolling(long_window).mean()
    signal = (short_average > long_average).astype(float)

    # No signal is permitted until both averages have enough observations.
    signal[long_average.isna()] = 0.0
    return signal


def performance_statistics(
    daily_returns: pd.Series, trading_days_per_year: int = 252
) -> dict[str, float]:
    """Turn daily portfolio returns into standard return/risk measurements."""

    returns = daily_returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("At least one return is needed to calculate statistics.")

    wealth = (1.0 + returns).cumprod()
    total_return = wealth.iloc[-1] - 1.0
    years = len(returns) / trading_days_per_year
    annualised_return = wealth.iloc[-1] ** (1.0 / years) - 1.0
    annualised_volatility = returns.std(ddof=1) * np.sqrt(trading_days_per_year)
    sharpe_ratio = (
        returns.mean() * trading_days_per_year / annualised_volatility
        if annualised_volatility > 0
        else np.nan
    )
    drawdown = wealth / wealth.cummax() - 1.0

    return {
        "total_return": float(total_return),
        "annualised_return": float(annualised_return),
        "annualised_volatility": float(annualised_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "maximum_drawdown": float(-drawdown.min()),
    }


def backtest(
    prices: pd.DataFrame,
    short_window: int,
    long_window: int,
    transaction_cost_bps: float = 5.0,
    trading_days_per_year: int = 252,
) -> dict[str, object]:
    """Simulate the strategy and an equal-weight buy-and-hold benchmark.

    Each active ETF receives 1/N of the portfolio, where N is the total number
    of ETFs.  Any unused fraction stays as cash earning 0%.  A signal calculated
    using today's close is shifted one row, so it can only affect tomorrow's
    return.  That shift is what prevents look-ahead bias.
    """

    prices = validate_prices(prices)
    asset_returns = prices.pct_change().fillna(0.0)
    signal = moving_average_signal(prices, short_window, long_window)
    number_of_assets = prices.shape[1]

    weights = signal.shift(1).fillna(0.0) / number_of_assets
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost_rate = transaction_cost_bps / 10_000.0
    costs = turnover * cost_rate
    strategy_returns = (weights * asset_returns).sum(axis=1) - costs

    # Buy equal monetary amounts once, then keep the resulting number of shares.
    # Taking the mean of normalised prices gives that fixed-share portfolio's
    # value.  Its weights naturally drift as some ETFs rise faster than others.
    gross_benchmark_wealth = (prices / prices.iloc[0]).mean(axis=1)
    benchmark_returns = gross_benchmark_wealth.pct_change().fillna(0.0)
    benchmark_returns.iloc[0] -= cost_rate  # cost of the initial purchase

    strategy_stats = performance_statistics(strategy_returns, trading_days_per_year)
    benchmark_stats = performance_statistics(benchmark_returns, trading_days_per_year)
    strategy_stats["annualised_outperformance"] = (
        strategy_stats["annualised_return"] - benchmark_stats["annualised_return"]
    )

    return {
        "signal": signal,
        "weights": weights,
        "turnover": turnover,
        "costs": costs,
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "strategy_wealth": (1.0 + strategy_returns).cumprod(),
        "benchmark_wealth": (1.0 + benchmark_returns).cumprod(),
        "strategy_stats": strategy_stats,
        "benchmark_stats": benchmark_stats,
    }


# ---------------------------------------------------------------------------
# 4. ROBUSTNESS CHECKS
# ---------------------------------------------------------------------------


def regime_analysis(
    result: dict[str, object], regimes: dict[str, tuple[str, str]] = REGIMES
) -> pd.DataFrame:
    """Calculate strategy statistics separately inside three historical periods."""

    strategy_returns = result["strategy_returns"]
    benchmark_returns = result["benchmark_returns"]
    rows = []
    for name, (start, end) in regimes.items():
        strategy_part = strategy_returns.loc[start:end]
        benchmark_part = benchmark_returns.loc[start:end]
        if strategy_part.empty:
            continue
        strategy_stats = performance_statistics(strategy_part)
        benchmark_stats = performance_statistics(benchmark_part)
        rows.append(
            {
                "regime": name,
                "start": strategy_part.index.min().date(),
                "end": strategy_part.index.max().date(),
                "annualised_return": strategy_stats["annualised_return"],
                "sharpe_ratio": strategy_stats["sharpe_ratio"],
                "maximum_drawdown": strategy_stats["maximum_drawdown"],
                "benchmark_annualised_return": benchmark_stats["annualised_return"],
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def parameter_stress_test(
    prices: pd.DataFrame,
    short_windows: tuple[int, ...] = SHORT_WINDOWS,
    long_windows: tuple[int, ...] = LONG_WINDOWS,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Backtest every valid short/long moving-average pair."""

    rows = []
    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue
            result = backtest(
                prices, short_window, long_window, transaction_cost_bps
            )
            stats = result["strategy_stats"]
            rows.append(
                {
                    "short_window": short_window,
                    "long_window": long_window,
                    **stats,
                }
            )
    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)


def walk_forward_validation(
    prices: pd.DataFrame,
    short_windows: tuple[int, ...] = SHORT_WINDOWS,
    long_windows: tuple[int, ...] = LONG_WINDOWS,
    transaction_cost_bps: float = 5.0,
    training_years: int = 5,
    testing_years: int = 1,
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    """Choose parameters on past data, then judge them on unseen future data.

    The training window rolls forward by one testing period.  Parameter choices
    maximise training Sharpe ratio.  Only the following test window contributes
    to the final out-of-sample result.
    """

    training_rows = training_years * trading_days_per_year
    testing_rows = testing_years * trading_days_per_year
    all_parameter_results: dict[tuple[int, int], dict[str, object]] = {}

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window < long_window:
                result = backtest(
                    prices, short_window, long_window, transaction_cost_bps
                )
                all_parameter_results[(short_window, long_window)] = result

    fold_rows = []
    test_return_parts = []
    previous_live_weights = pd.Series(0.0, index=prices.columns)
    asset_returns = prices.pct_change().fillna(0.0)
    cost_rate = transaction_cost_bps / 10_000.0
    start_row = 0
    fold_number = 1

    while start_row + training_rows + testing_rows <= len(prices):
        train_slice = slice(start_row, start_row + training_rows)
        test_slice = slice(
            start_row + training_rows,
            start_row + training_rows + testing_rows,
        )

        best_parameters = None
        best_training_sharpe = -np.inf
        for parameters, result in all_parameter_results.items():
            training_returns = result["strategy_returns"].iloc[train_slice]
            stats = performance_statistics(training_returns)
            sharpe = stats["sharpe_ratio"]
            if np.isfinite(sharpe) and sharpe > best_training_sharpe:
                best_parameters = parameters
                best_training_sharpe = sharpe

        if best_parameters is None:
            raise RuntimeError("No valid parameter pair was found in a training fold.")

        chosen_weights = all_parameter_results[best_parameters]["weights"].iloc[
            test_slice
        ]
        chosen_asset_returns = asset_returns.iloc[test_slice]
        live_turnover = chosen_weights.diff().abs().sum(axis=1)
        live_turnover.iloc[0] = (
            chosen_weights.iloc[0] - previous_live_weights
        ).abs().sum()
        chosen_returns = (
            chosen_weights * chosen_asset_returns
        ).sum(axis=1) - live_turnover * cost_rate
        previous_live_weights = chosen_weights.iloc[-1]
        test_stats = performance_statistics(chosen_returns)
        test_return_parts.append(chosen_returns)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_start": prices.index[train_slice.start].date(),
                "train_end": prices.index[train_slice.stop - 1].date(),
                "test_start": prices.index[test_slice.start].date(),
                "test_end": prices.index[test_slice.stop - 1].date(),
                "chosen_short_window": best_parameters[0],
                "chosen_long_window": best_parameters[1],
                "training_sharpe": best_training_sharpe,
                "test_annualised_return": test_stats["annualised_return"],
                "test_sharpe": test_stats["sharpe_ratio"],
                "test_maximum_drawdown": test_stats["maximum_drawdown"],
            }
        )
        fold_number += 1
        start_row += testing_rows

    if not test_return_parts:
        raise ValueError("There is not enough data for one train/test fold.")

    out_of_sample_returns = pd.concat(test_return_parts)
    overall_stats = performance_statistics(out_of_sample_returns)
    return pd.DataFrame(fold_rows).set_index("fold"), out_of_sample_returns, overall_stats


# ---------------------------------------------------------------------------
# 5. OUTPUT
# ---------------------------------------------------------------------------


def save_chart(result: dict[str, object], output_path: Path) -> None:
    """Save a chart of how one pound grows under each approach."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chart skipped because matplotlib is not installed.")
        return

    figure, axis = plt.subplots(figsize=(11, 6))
    result["strategy_wealth"].plot(ax=axis, label="Momentum strategy", linewidth=2)
    result["benchmark_wealth"].plot(ax=axis, label="Equal-weight buy and hold")
    axis.set(title="Growth of £1", xlabel="Date", ylabel="Portfolio value (£)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def percent(value: float) -> str:
    """Format a decimal such as 0.143 as the human-friendly string 14.30%."""

    return f"{100 * value:.2f}%"


def print_summary(
    result: dict[str, object], stress_results: pd.DataFrame, walk_stats: dict[str, float]
) -> None:
    """Print the headline results with consistent formatting."""

    strategy = result["strategy_stats"]
    benchmark = result["benchmark_stats"]
    print("\nMAIN BACKTEST")
    print(f"Strategy annualised return: {percent(strategy['annualised_return'])}")
    print(f"Strategy Sharpe ratio:      {strategy['sharpe_ratio']:.2f}")
    print(f"Strategy maximum drawdown:  {percent(strategy['maximum_drawdown'])}")
    print(f"Benchmark annualised return:{percent(benchmark['annualised_return']):>10}")
    print(f"Annualised outperformance:  {percent(strategy['annualised_outperformance'])}")
    print(f"\nParameter combinations tested: {len(stress_results)}")
    print("WALK-FORWARD OUT-OF-SAMPLE")
    print(f"Annualised return: {percent(walk_stats['annualised_return'])}")
    print(f"Sharpe ratio:      {walk_stats['sharpe_ratio']:.2f}")
    print(f"Maximum drawdown:  {percent(walk_stats['maximum_drawdown'])}")


def parse_arguments() -> argparse.Namespace:
    """Read the user's command-line choices."""

    parser = argparse.ArgumentParser(description=__doc__)
    data_choice = parser.add_mutually_exclusive_group(required=True)
    data_choice.add_argument(
        "--demo", action="store_true", help="use reproducible artificial prices"
    )
    data_choice.add_argument(
        "--real-data", action="store_true", help="download historical ETF prices"
    )
    parser.add_argument("--short", type=int, default=50, help="short average window")
    parser.add_argument("--long", type=int, default=200, help="long average window")
    parser.add_argument(
        "--cost-bps", type=float, default=5.0, help="cost per trade in basis points"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="results folder"
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete research process from data to saved results."""

    arguments = parse_arguments()
    settings = Settings(
        short_window=arguments.short,
        long_window=arguments.long,
        transaction_cost_bps=arguments.cost_bps,
    )
    output_directory = arguments.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    prices = make_demo_prices(settings) if arguments.demo else download_adjusted_prices(settings)
    prices.to_csv(output_directory / "prices.csv")

    result = backtest(
        prices,
        settings.short_window,
        settings.long_window,
        settings.transaction_cost_bps,
    )
    regimes = regime_analysis(result)
    stress_results = parameter_stress_test(
        prices, transaction_cost_bps=settings.transaction_cost_bps
    )
    folds, out_of_sample_returns, walk_stats = walk_forward_validation(
        prices, transaction_cost_bps=settings.transaction_cost_bps
    )

    daily_results = pd.DataFrame(
        {
            "strategy_return": result["strategy_returns"],
            "benchmark_return": result["benchmark_returns"],
            "strategy_wealth": result["strategy_wealth"],
            "benchmark_wealth": result["benchmark_wealth"],
            "turnover": result["turnover"],
            "transaction_cost": result["costs"],
        }
    )
    daily_results.to_csv(output_directory / "daily_backtest.csv")
    regimes.to_csv(output_directory / "regime_results.csv")
    stress_results.to_csv(output_directory / "parameter_stress_test.csv", index=False)
    folds.to_csv(output_directory / "walk_forward_folds.csv")
    out_of_sample_returns.rename("out_of_sample_return").to_csv(
        output_directory / "walk_forward_returns.csv"
    )
    save_chart(result, output_directory / "wealth_chart.png")
    print_summary(result, stress_results, walk_stats)
    print(f"\nDetailed results saved in: {output_directory.resolve()}")


if __name__ == "__main__":
    main()
