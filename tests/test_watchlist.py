# Copyright 2026 diywallst
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watchlist batch runner."""

import csv
from pathlib import Path

from tradingagents.watchlist import (
    WatchlistResult,
    normalize_tickers,
    rank_results,
    run_watchlist,
    write_summary,
)


class FakeGraph:
    def __init__(self, signals):
        self.signals = signals
        self.calls = []
        self.saved = []

    def propagate(self, ticker, trade_date, asset_type="stock"):
        self.calls.append((ticker, trade_date, asset_type))
        signal = self.signals[ticker]
        if isinstance(signal, Exception):
            raise signal
        return {"ticker": ticker}, signal

    def save_reports(self, final_state, ticker, save_path=None):
        self.saved.append(ticker)
        return Path("/reports") / ticker / "complete_report.md"


def test_normalize_tickers_splits_uppercases_and_dedupes():
    assert normalize_tickers(["nvda, aapl", "NVDA  msft", ""]) == ["NVDA", "AAPL", "MSFT"]


def test_run_continues_past_failures():
    graph = FakeGraph({"NVDA": "Buy", "BAD": ValueError("no data"), "TSLA": "Sell"})
    seen = []
    results = run_watchlist(
        graph, ["nvda", "bad", "tsla"], "2026-09-11", on_result=lambda r: seen.append(r.ticker)
    )

    assert [r.ticker for r in results] == ["NVDA", "BAD", "TSLA"]
    assert seen == ["NVDA", "BAD", "TSLA"]
    assert results[0].signal == "Buy" and results[0].ok
    assert not results[1].ok
    assert results[1].error == "ValueError: no data"
    assert results[2].signal == "Sell"


def test_asset_type_callable_and_report_saving():
    graph = FakeGraph({"BTC-USD": "Hold", "NVDA": "Buy"})
    results = run_watchlist(
        graph,
        ["BTC-USD", "NVDA"],
        "2026-09-11",
        asset_type=lambda t: "crypto" if t.endswith("-USD") else "stock",
        save_reports=True,
    )
    assert graph.calls == [("BTC-USD", "2026-09-11", "crypto"), ("NVDA", "2026-09-11", "stock")]
    assert graph.saved == ["BTC-USD", "NVDA"]
    assert results[1].report_path.endswith("NVDA/complete_report.md")


def test_rank_bullish_first_then_review_then_errors():
    results = [
        WatchlistResult("E", None, error="boom"),
        WatchlistResult("S", "Sell"),
        WatchlistResult("R", "REVIEW"),
        WatchlistResult("B", "Buy"),
        WatchlistResult("H", "Hold"),
    ]
    assert [r.ticker for r in rank_results(results)] == ["B", "H", "S", "R", "E"]


def test_write_summary(tmp_path):
    results = [
        WatchlistResult("TSLA", "Underweight"),
        WatchlistResult("BAD", None, error="KeyError: 'x|y'"),
        WatchlistResult("NVDA", "Overweight"),
    ]
    csv_path, md_path = write_summary(results, "2026-09-11", tmp_path / "out")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["ticker"] for r in rows] == ["NVDA", "TSLA", "BAD"]
    assert rows[2]["error"] == "KeyError: 'x|y'"

    md = md_path.read_text(encoding="utf-8")
    assert "# Watchlist signals — 2026-09-11" in md
    assert "| BAD | ERROR | KeyError: 'x\\|y' |" in md
    assert "2 analyzed · 1 failed" in md
