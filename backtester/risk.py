"""Pre-trade portfolio constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskManager:
    max_position_weight: float = 1.0
    max_gross_exposure: float = 1.0
    min_cash_weight: float = 0.0
    long_only: bool = True

    def validate_targets(
        self, target_weights: Mapping[str, float], allowed_tickers: tuple[str, ...]
    ) -> dict[str, float]:
        unknown = set(target_weights) - set(allowed_tickers)
        if unknown:
            raise ValueError(f"Targets contain unknown tickers: {sorted(unknown)}")

        cleaned = {ticker: float(target_weights.get(ticker, 0.0)) for ticker in allowed_tickers}
        if self.long_only and any(weight < -1e-12 for weight in cleaned.values()):
            raise ValueError("Negative target weights are forbidden in long-only mode.")
        if any(abs(weight) > self.max_position_weight + 1e-12 for weight in cleaned.values()):
            raise ValueError("A target exceeds the maximum position weight.")
        gross = sum(abs(weight) for weight in cleaned.values())
        maximum_invested = min(self.max_gross_exposure, 1.0 - self.min_cash_weight)
        if gross > maximum_invested + 1e-12:
            raise ValueError("Target weights exceed the permitted gross exposure.")
        return cleaned

