# Copyright 2026 diywallst
# SPDX-License-Identifier: Apache-2.0
"""Tests for the decision-log performance scorecard."""

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.scorecard import build_scorecard, load_scorecard, render_markdown


def _entry(ticker, rating, raw, alpha, pending=False):
    return {"ticker": ticker, "rating": rating, "raw": raw, "alpha": alpha, "pending": pending}


ENTRIES = [
    _entry("NVDA", "Buy", "+2.0%", "+1.0%"),  # long, beat benchmark: hit
    _entry("NVDA", "Sell", "-3.0%", "-2.0%"),  # short, lagged benchmark: hit
    _entry("AAPL", "Overweight", "+0.5%", "-1.0%"),  # long, lagged: miss
    _entry("AAPL", "Hold", "+0.5%", "+0.5%"),  # neutral: returns only
    _entry("MSFT", "Buy", None, None, pending=True),
    _entry("MSFT", "Maybe", "+1.0%", "+1.0%"),  # unknown rating: unscored
]


def test_overall_stats():
    card = build_scorecard(ENTRIES)
    assert card.overall.count == 4
    assert card.overall.directional == 3
    assert card.overall.hits == 2
    assert card.overall.hit_rate == pytest.approx(2 / 3)
    assert card.overall.avg_raw == pytest.approx(0.0)
    assert card.overall.avg_alpha == pytest.approx((0.01 - 0.02 - 0.01 + 0.005) / 4)
    assert card.overall.avg_signed_alpha == pytest.approx((0.01 + 0.02 - 0.01) / 3)
    assert card.pending == 1
    assert card.skipped == 1


def test_breakdowns():
    card = build_scorecard(ENTRIES)
    assert card.by_ticker["NVDA"].hit_rate == 1.0
    assert card.by_ticker["AAPL"].hit_rate == 0.0
    assert card.by_rating["Hold"].directional == 0
    assert card.by_rating["Hold"].hit_rate is None
    assert card.by_rating["Hold"].avg_signed_alpha is None


def test_load_from_real_log(tmp_path):
    path = tmp_path / "memory.md"
    log = TradingMemoryLog({"memory_log_path": str(path)})
    log.store_decision("TSLA", "2026-01-05", "Rating: Underweight\nTrim.")
    log.update_with_outcome("TSLA", "2026-01-05", -0.04, -0.03, 5, "Good call.")
    log.store_decision("TSLA", "2026-01-12", "Rating: Buy\nAdd.")

    card = load_scorecard(path)
    assert card.overall.count == 1
    assert card.overall.hits == 1
    assert card.by_rating["Underweight"].avg_alpha == pytest.approx(-0.03)
    assert card.pending == 1


def test_missing_log_renders_empty(tmp_path):
    card = load_scorecard(tmp_path / "nope.md")
    assert card.overall.count == 0
    assert "No resolved decisions yet." in render_markdown(card)


def test_render_orders_ratings_bullish_first():
    text = render_markdown(build_scorecard(ENTRIES))
    assert text.index("| Buy |") < text.index("| Overweight |") < text.index("| Hold |")
    assert text.index("| Hold |") < text.index("| Sell |")
    assert "Pending: 1 · Unscored: 1" in text
