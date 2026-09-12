# Copyright 2026 diywallst
# SPDX-License-Identifier: Apache-2.0
"""Run the agent pipeline over a watchlist and summarize the signals.

``run_watchlist`` calls ``TradingAgentsGraph.propagate`` once per ticker. A
failure on one ticker (bad symbol, provider error, missing data) is recorded
and the batch moves on, so an overnight run over 30 names doesn't die on the
fourth. ``write_summary`` writes a CSV and a markdown table ranked from most
bullish to most bearish.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from tradingagents.agents.utils.rating import RATINGS_5_TIER
from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)


@dataclass
class WatchlistResult:
    ticker: str
    signal: str | None
    error: str | None = None
    report_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Split comma/space separated input, uppercase, and de-duplicate in order."""
    seen: dict[str, None] = {}
    for item in tickers:
        for token in re.split(r"[\s,]+", item or ""):
            token = token.strip().upper()
            if token:
                seen.setdefault(token, None)
    return list(seen)


def run_watchlist(
    graph,
    tickers: Iterable[str],
    trade_date: str,
    *,
    asset_type: str | Callable[[str], str] = "stock",
    save_reports: bool = False,
    on_result: Callable[[WatchlistResult], None] | None = None,
) -> list[WatchlistResult]:
    """Analyze each ticker with ``graph`` and return one result per ticker.

    ``asset_type`` is either a fixed value or a callable mapping ticker to asset
    type (the CLI passes ``cli.utils.detect_asset_type``). ``on_result`` is
    called after each ticker, for progress output.
    """
    results = []
    for ticker in normalize_tickers(tickers):
        kind = asset_type(ticker) if callable(asset_type) else asset_type
        try:
            final_state, signal = graph.propagate(ticker, trade_date, asset_type=kind)
            report_path = str(graph.save_reports(final_state, ticker)) if save_reports else None
            result = WatchlistResult(ticker, signal, report_path=report_path)
        except Exception as exc:  # one bad ticker must not abort the batch
            logger.warning("Watchlist run failed for %s: %s", ticker, exc)
            result = WatchlistResult(ticker, None, error=f"{type(exc).__name__}: {exc}")
        if on_result is not None:
            on_result(result)
        results.append(result)
    return results


def _rank(result: WatchlistResult) -> int:
    if not result.ok:
        return len(RATINGS_5_TIER) + 1
    if result.signal in RATINGS_5_TIER:
        return RATINGS_5_TIER.index(result.signal)
    return len(RATINGS_5_TIER)  # REVIEW or anything unrecognized


def rank_results(results: list[WatchlistResult]) -> list[WatchlistResult]:
    """Order results most bullish first, then REVIEW, then failures (stable)."""
    return sorted(results, key=_rank)


def render_summary_markdown(results: list[WatchlistResult], trade_date: str) -> str:
    lines = [
        f"# Watchlist signals — {trade_date}",
        "",
        "| Ticker | Signal | Notes |",
        "|---|---|---|",
    ]
    for r in rank_results(results):
        signal = r.signal if r.ok else "ERROR"
        notes = (r.error or r.report_path or "").replace("|", "\\|")
        lines.append(f"| {r.ticker} | {signal} | {notes} |")
    failed = sum(1 for r in results if not r.ok)
    lines += ["", f"{len(results) - failed} analyzed · {failed} failed"]
    return "\n".join(lines) + "\n"


def write_summary(
    results: list[WatchlistResult], trade_date: str, out_dir: str | Path
) -> tuple[Path, Path]:
    """Write ``watchlist_<date>.csv`` and ``.md`` under ``out_dir``; return both paths."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"watchlist_{safe_ticker_component(str(trade_date))}"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "signal", "error", "report_path"])
        for r in rank_results(results):
            writer.writerow([r.ticker, r.signal or "", r.error or "", r.report_path or ""])
    md_path.write_text(render_summary_markdown(results, trade_date), encoding="utf-8")
    return csv_path, md_path
