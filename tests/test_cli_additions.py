# Copyright 2026 diywallst
# SPDX-License-Identifier: Apache-2.0
"""CLI tests for the scorecard, size, and batch commands."""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import cli.main as cli_main
from tradingagents.agents.utils.memory import TradingMemoryLog

runner = CliRunner()


def test_bare_invocation_still_runs_analysis(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "run_analysis", lambda checkpoint=None: calls.append(checkpoint))

    assert runner.invoke(cli_main.app, []).exit_code == 0
    assert runner.invoke(cli_main.app, ["--checkpoint"]).exit_code == 0
    assert runner.invoke(cli_main.app, ["analyze", "--no-checkpoint"]).exit_code == 0
    assert calls == [None, True, False]


def test_scorecard_reads_given_log(tmp_path):
    path = tmp_path / "memory.md"
    log = TradingMemoryLog({"memory_log_path": str(path)})
    log.store_decision("NVDA", "2026-01-05", "Rating: Buy\nAdd.")
    log.update_with_outcome("NVDA", "2026-01-05", 0.03, 0.02, 5, "Worked.")

    result = runner.invoke(cli_main.app, ["scorecard", "--log", str(path)])
    assert result.exit_code == 0, result.output
    assert "NVDA" in result.output
    assert "100%" in result.output


def test_size_uses_loaded_prices(monkeypatch):
    prices = pd.DataFrame({"High": [101.0] * 20, "Low": [99.0] * 20, "Close": [100.0] * 20})
    calls = []

    def fake_load(symbol, curr_date):
        calls.append((symbol, curr_date))
        return prices

    monkeypatch.setattr("tradingagents.dataflows.stockstats_utils.load_ohlcv", fake_load)
    result = runner.invoke(
        cli_main.app,
        ["size", "nvda", "--rating", "Buy", "--portfolio", "100000", "--date", "2026-09-11"],
    )
    assert result.exit_code == 0, result.output
    assert calls == [("NVDA", "2026-09-11")]
    assert "long" in result.output
    assert "100" in result.output


def test_size_rejects_bad_date():
    result = runner.invoke(
        cli_main.app, ["size", "NVDA", "--rating", "Buy", "--portfolio", "1000", "--date", "9/11"]
    )
    assert result.exit_code == 2


class FakeGraph:
    instances: list = []

    def __init__(self, selected_analysts=(), debug=False, config=None, callbacks=None):
        self.selected_analysts = list(selected_analysts)
        FakeGraph.instances.append(self)

    def propagate(self, ticker, trade_date, asset_type="stock"):
        if ticker == "BAD":
            raise RuntimeError("no data")
        return {}, "Buy"

    def save_reports(self, final_state, ticker, save_path=None):
        return Path("reports") / ticker


def test_batch_writes_summary_and_skips_fundamentals_for_crypto(monkeypatch, tmp_path):
    FakeGraph.instances = []
    monkeypatch.setattr(cli_main, "TradingAgentsGraph", FakeGraph)

    result = runner.invoke(
        cli_main.app,
        ["batch", "NVDA,BTC-USD", "BAD", "--date", "2026-09-11", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "watchlist_2026-09-11.csv").exists()
    md = (tmp_path / "watchlist_2026-09-11.md").read_text(encoding="utf-8")
    assert "| BAD | ERROR | RuntimeError: no data |" in md

    analyst_sets = sorted(g.selected_analysts for g in FakeGraph.instances)
    assert analyst_sets == [
        ["market", "social", "news"],
        ["market", "social", "news", "fundamentals"],
    ]


def test_batch_exits_nonzero_when_everything_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_main, "TradingAgentsGraph", FakeGraph)
    result = runner.invoke(cli_main.app, ["batch", "BAD", "--date", "2026-09-11", "--out", str(tmp_path)])
    assert result.exit_code == 1
