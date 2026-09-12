# Copyright 2026 diywallst
# SPDX-License-Identifier: Apache-2.0
"""Tests for ATR-based position sizing."""

import pandas as pd
import pytest

from tradingagents.sizing import average_true_range, suggest_position


def _flat_prices(bars=20, close=100.0, spread=1.0):
    """Constant close with a fixed high/low band: true range == 2 * spread."""
    return pd.DataFrame(
        {
            "High": [close + spread] * bars,
            "Low": [close - spread] * bars,
            "Close": [close] * bars,
        }
    )


def test_atr_of_flat_band():
    assert average_true_range(_flat_prices(), 14) == pytest.approx(2.0)


def test_atr_uses_gaps_from_previous_close():
    prices = pd.DataFrame({"High": [101, 111], "Low": [99, 109], "Close": [100, 110]})
    # Second bar gaps up: |high - prev close| = 11 dominates high - low = 2.
    assert average_true_range(prices, 1) == pytest.approx(11.0)


def test_atr_needs_enough_bars():
    with pytest.raises(ValueError, match="at least 15 bars"):
        average_true_range(_flat_prices(bars=10), 14)


def test_atr_needs_ohlc_columns():
    with pytest.raises(ValueError, match="missing columns"):
        average_true_range(pd.DataFrame({"Close": [1.0] * 20}), 14)


def test_buy_is_capped_by_max_position():
    s = suggest_position("Buy", _flat_prices(), 100_000)
    # Risk sizing: $1,000 / (2 ATR = $4) = 250 shares; 10% cap = 100 shares.
    assert s.action == "long"
    assert s.shares == 100
    assert s.capped
    assert s.weight == pytest.approx(0.10)
    assert s.stop_price == pytest.approx(96.0)
    assert s.risk_amount == pytest.approx(400.0)


def test_buy_risk_sized_when_cap_is_loose():
    s = suggest_position("Buy", _flat_prices(), 100_000, max_position=1.0)
    assert s.shares == 250
    assert not s.capped
    assert s.risk_amount == pytest.approx(1_000.0)


def test_overweight_uses_half_conviction():
    s = suggest_position("overweight", _flat_prices(), 100_000, max_position=1.0)
    assert s.rating == "Overweight"
    assert s.shares == 125


def test_sell_exits_on_long_only_book():
    s = suggest_position("Sell", _flat_prices(), 100_000)
    assert s.action == "exit"
    assert s.shares == 0


def test_sell_shorts_when_allowed():
    s = suggest_position("Sell", _flat_prices(), 100_000, max_position=1.0, allow_short=True)
    assert s.action == "short"
    assert s.shares == -250
    assert s.weight == pytest.approx(-0.25)
    assert s.stop_price == pytest.approx(104.0)


@pytest.mark.parametrize("rating, action", [("Hold", "none"), ("REVIEW", "review"), ("", "review")])
def test_non_tradeable_ratings(rating, action):
    s = suggest_position(rating, _flat_prices(), 100_000)
    assert s.action == action
    assert s.shares == 0


def test_tiny_portfolio_yields_zero_shares_with_note():
    s = suggest_position("Buy", _flat_prices(close=1_000.0, spread=50.0), 1_000)
    assert s.shares == 0
    assert "smaller than one share" in s.note


@pytest.mark.parametrize(
    "kwargs",
    [
        {"portfolio_value": 0},
        {"portfolio_value": 1_000, "risk_per_trade": 0},
        {"portfolio_value": 1_000, "max_position": 1.5},
        {"portfolio_value": 1_000, "atr_multiple": 0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        suggest_position("Buy", _flat_prices(), **kwargs)
