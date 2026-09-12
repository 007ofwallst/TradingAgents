# Copyright 2026 diywallst
# SPDX-License-Identifier: Apache-2.0
"""Volatility-aware position sizing for a 5-tier rating.

The agents decide *direction and conviction*; this module turns that into a
concrete share count using fixed-fractional risk:

1. Conviction scales the risk budget: Buy/Sell use the full ``risk_per_trade``,
   Overweight/Underweight half of it, Hold none.
2. The stop sits ``atr_multiple`` Average True Ranges from the latest close, so
   volatile names get smaller positions for the same dollar risk.
3. ``shares = risk_budget / stop_distance``, capped so the position never
   exceeds ``max_position`` (also scaled by conviction) of the portfolio.

Bearish ratings produce an ``exit`` action unless ``allow_short`` is set. A
``REVIEW`` signal or unknown rating never produces a position.

This is a mechanical risk calculation, not investment advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from tradingagents.agents.utils.rating import RATINGS_5_TIER, is_review

RATING_CONVICTION: dict[str, float] = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": -0.5,
    "Sell": -1.0,
}


@dataclass(frozen=True)
class PositionSuggestion:
    rating: str
    action: str  # "long", "short", "exit", "none", or "review"
    shares: int  # signed: negative for a short
    weight: float  # signed fraction of portfolio value
    entry_price: float | None = None
    stop_price: float | None = None
    atr: float | None = None
    risk_amount: float = 0.0  # dollars lost if the stop is hit
    capped: bool = False  # True when max_position bound the size
    note: str = ""


def average_true_range(prices: pd.DataFrame, period: int = 14) -> float:
    """Simple-moving-average ATR over the last ``period`` bars of an OHLC frame."""
    missing = {"High", "Low", "Close"} - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing columns: {sorted(missing)}")
    if len(prices) < period + 1:
        raise ValueError(f"need at least {period + 1} bars for ATR({period}), got {len(prices)}")
    prev_close = prices["Close"].shift(1)
    true_range = pd.concat(
        [
            prices["High"] - prices["Low"],
            (prices["High"] - prev_close).abs(),
            (prices["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.iloc[-period:].mean())
    if not math.isfinite(atr) or atr <= 0:
        raise ValueError("ATR is not positive; price data may be flat or incomplete")
    return atr


def _normalize_rating(rating: str) -> str:
    cleaned = (rating or "").strip()
    for canonical in RATINGS_5_TIER:
        if cleaned.lower() == canonical.lower():
            return canonical
    return cleaned


def suggest_position(
    rating: str,
    prices: pd.DataFrame,
    portfolio_value: float,
    *,
    risk_per_trade: float = 0.01,
    atr_period: int = 14,
    atr_multiple: float = 2.0,
    max_position: float = 0.10,
    allow_short: bool = False,
) -> PositionSuggestion:
    """Size a position for ``rating`` given recent OHLC ``prices``.

    ``prices`` must be sorted oldest to newest and end at the decision date
    (e.g. the frame from ``dataflows.stockstats_utils.load_ohlcv``).
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    if not 0 < risk_per_trade <= 1:
        raise ValueError("risk_per_trade must be in (0, 1]")
    if not 0 < max_position <= 1:
        raise ValueError("max_position must be in (0, 1]")
    if atr_multiple <= 0:
        raise ValueError("atr_multiple must be positive")

    rating = _normalize_rating(rating)
    if is_review(rating) or rating not in RATING_CONVICTION:
        return PositionSuggestion(
            rating, "review", 0, 0.0,
            note="No tradeable rating; re-run or review the decision manually.",
        )

    conviction = RATING_CONVICTION[rating]
    if conviction == 0:
        return PositionSuggestion(rating, "none", 0, 0.0, note="Hold: no new position.")
    if conviction < 0 and not allow_short:
        return PositionSuggestion(
            rating, "exit", 0, 0.0,
            note="Bearish rating on a long-only book: reduce or exit any existing position.",
        )

    price = float(prices["Close"].iloc[-1])
    if not math.isfinite(price) or price <= 0:
        raise ValueError("latest close must be a positive number")
    atr = average_true_range(prices, atr_period)
    stop_distance = atr * atr_multiple

    strength = abs(conviction)
    risk_shares = math.floor(portfolio_value * risk_per_trade * strength / stop_distance)
    cap_shares = math.floor(portfolio_value * max_position * strength / price)
    shares = min(risk_shares, cap_shares)
    capped = cap_shares < risk_shares

    sign = 1 if conviction > 0 else -1
    note = ""
    if shares == 0:
        note = "Risk budget is smaller than one share at this stop distance."
    elif capped:
        note = f"Capped at {max_position * strength:.1%} of portfolio."

    return PositionSuggestion(
        rating=rating,
        action="long" if sign > 0 else "short",
        shares=sign * shares,
        weight=sign * shares * price / portfolio_value,
        entry_price=price,
        stop_price=price - sign * stop_distance,
        atr=atr,
        risk_amount=shares * stop_distance,
        capped=capped,
        note=note,
    )
