"""Performance, risk and trading statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    equity: pd.Series,
    fills: pd.DataFrame | None = None,
    trading_days_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> dict[str, float]:
    values = equity.dropna().astype(float)
    if len(values) < 2:
        raise ValueError("At least two equity observations are required.")
    if (values <= 0).any():
        raise ValueError("Equity must remain positive for return statistics.")

    returns = values.pct_change().dropna()
    elapsed_years = len(returns) / trading_days_per_year
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    annualised_return = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
    annualised_volatility = returns.std(ddof=1) * np.sqrt(trading_days_per_year)
    daily_risk_free = (1.0 + annual_risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess = returns - daily_risk_free
    sharpe = (
        excess.mean() / returns.std(ddof=1) * np.sqrt(trading_days_per_year)
        if returns.std(ddof=1) > 0
        else np.nan
    )
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    downside = returns[returns < 0].std(ddof=1) * np.sqrt(trading_days_per_year)
    sortino = (
        excess.mean() * trading_days_per_year / downside
        if pd.notna(downside) and downside > 0
        else np.nan
    )
    calmar = annualised_return / (-drawdown.min()) if drawdown.min() < 0 else np.nan

    metrics = {
        "total_return": float(total_return),
        "annualised_return": float(annualised_return),
        "annualised_volatility": float(annualised_volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "maximum_drawdown": float(-drawdown.min()),
        "calmar_ratio": float(calmar),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "positive_day_fraction": float((returns > 0).mean()),
    }
    if fills is not None and not fills.empty:
        metrics.update(
            {
                "fill_count": float(len(fills)),
                "traded_notional": float(fills["notional"].abs().sum()),
                "total_commission": float(fills["commission"].sum()),
                "estimated_spread_cost": float(fills["spread_cost"].sum()),
                "estimated_slippage_cost": float(fills["slippage_cost"].sum()),
            }
        )
    else:
        metrics.update(
            {
                "fill_count": 0.0,
                "traded_notional": 0.0,
                "total_commission": 0.0,
                "estimated_spread_cost": 0.0,
                "estimated_slippage_cost": 0.0,
            }
        )
    return metrics


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    monthly = equity.resample("ME").last().pct_change().dropna()
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="return")


def yearly_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("YE").last().pct_change().dropna().rename("return")

