"""Commission, spread and slippage assumptions."""

from __future__ import annotations

from dataclasses import dataclass

from .events import OrderSide


@dataclass(frozen=True)
class CostModel:
    commission_fixed: float = 0.0
    commission_bps: float = 1.0
    spread_bps: float = 3.0
    slippage_bps: float = 1.0

    def execution_price(self, reference_price: float, side: OrderSide) -> tuple[float, float, float]:
        """Return fill price and diagnostic spread/slippage cash costs per unit."""

        if reference_price <= 0:
            raise ValueError("Reference price must be positive.")
        direction = 1.0 if side == OrderSide.BUY else -1.0
        half_spread_rate = self.spread_bps / 2.0 / 10_000.0
        slippage_rate = self.slippage_bps / 10_000.0
        spread_per_unit = reference_price * half_spread_rate
        slippage_per_unit = reference_price * slippage_rate
        fill_price = reference_price * (
            1.0 + direction * (half_spread_rate + slippage_rate)
        )
        return fill_price, spread_per_unit, slippage_per_unit

    def commission(self, notional: float) -> float:
        return self.commission_fixed + abs(notional) * self.commission_bps / 10_000.0

