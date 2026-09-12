# Copyright 2026 diywallst
# SPDX-License-Identifier: Apache-2.0
"""Performance scorecard over the resolved decisions in the memory log.

Every completed run appends a pending entry to the decision log, and the next
same-ticker run settles it with the realized raw and benchmark-relative (alpha)
return. This module turns those settled entries into a track record: how often
directional calls beat the benchmark, and the average alpha earned per rating,
per ticker, and overall.

Scoring rules:
- Buy / Overweight imply a long bias (+1); Underweight / Sell a short bias (-1).
- A directional call is a *hit* when ``direction * alpha > 0``.
- Hold calls count toward returns but not toward hit rate or signed alpha,
  since they make no directional bet.
- Pending entries and entries with an unrecognized rating are reported as
  counts but not scored.

Returns are read back from the log tag, which stores them rounded to 0.1%, so
averages carry that precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tradingagents.agents.utils.memory import TradingMemoryLog

RATING_DIRECTION: dict[str, int] = {
    "Buy": 1,
    "Overweight": 1,
    "Hold": 0,
    "Underweight": -1,
    "Sell": -1,
}


def _parse_pct(value: str | None) -> float | None:
    """Parse a log-tag percentage such as ``+1.5%`` into a fraction (0.015)."""
    if not value:
        return None
    try:
        return float(value.strip().rstrip("%")) / 100
    except ValueError:
        return None


@dataclass
class Bucket:
    """Aggregated stats for one slice of decisions."""

    count: int = 0
    directional: int = 0
    hits: int = 0
    raw_sum: float = 0.0
    alpha_sum: float = 0.0
    signed_alpha_sum: float = 0.0

    def add(self, direction: int, raw: float, alpha: float) -> None:
        self.count += 1
        self.raw_sum += raw
        self.alpha_sum += alpha
        if direction:
            self.directional += 1
            self.signed_alpha_sum += direction * alpha
            if direction * alpha > 0:
                self.hits += 1

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.directional if self.directional else None

    @property
    def avg_raw(self) -> float | None:
        return self.raw_sum / self.count if self.count else None

    @property
    def avg_alpha(self) -> float | None:
        return self.alpha_sum / self.count if self.count else None

    @property
    def avg_signed_alpha(self) -> float | None:
        """Average alpha earned by following the call's direction."""
        return self.signed_alpha_sum / self.directional if self.directional else None


@dataclass
class Scorecard:
    overall: Bucket = field(default_factory=Bucket)
    by_rating: dict[str, Bucket] = field(default_factory=dict)
    by_ticker: dict[str, Bucket] = field(default_factory=dict)
    pending: int = 0
    skipped: int = 0


def build_scorecard(entries: list[dict]) -> Scorecard:
    """Score parsed memory-log entries (as returned by ``TradingMemoryLog.load_entries``)."""
    card = Scorecard()
    for entry in entries:
        if entry.get("pending"):
            card.pending += 1
            continue
        direction = RATING_DIRECTION.get(entry.get("rating", ""))
        raw = _parse_pct(entry.get("raw"))
        alpha = _parse_pct(entry.get("alpha"))
        if direction is None or raw is None or alpha is None:
            card.skipped += 1
            continue
        card.overall.add(direction, raw, alpha)
        card.by_rating.setdefault(entry["rating"], Bucket()).add(direction, raw, alpha)
        card.by_ticker.setdefault(entry["ticker"], Bucket()).add(direction, raw, alpha)
    return card


def load_scorecard(memory_log_path: str | Path) -> Scorecard:
    """Build a scorecard straight from a memory log file (empty if it does not exist)."""
    path = Path(memory_log_path).expanduser()
    if not path.exists():
        return Scorecard()
    return build_scorecard(TradingMemoryLog({"memory_log_path": str(path)}).load_entries())


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _table(label: str, rows: dict[str, Bucket]) -> list[str]:
    lines = [
        f"| {label} | Calls | Hit rate | Avg raw | Avg alpha | Avg signed alpha |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, b in rows.items():
        lines.append(
            f"| {name} | {b.count} | {_fmt_rate(b.hit_rate)} | {_fmt_pct(b.avg_raw)} "
            f"| {_fmt_pct(b.avg_alpha)} | {_fmt_pct(b.avg_signed_alpha)} |"
        )
    return lines


def render_markdown(card: Scorecard) -> str:
    """Render a scorecard as a markdown report."""
    lines = ["# TradingAgents Scorecard", ""]
    if not card.overall.count:
        lines.append("No resolved decisions yet.")
        if card.pending:
            lines.append(f"{card.pending} decision(s) pending resolution.")
        return "\n".join(lines) + "\n"

    lines += _table("Scope", {"All decisions": card.overall}) + [""]
    ordered = {r: card.by_rating[r] for r in RATING_DIRECTION if r in card.by_rating}
    lines += ["## By rating", ""] + _table("Rating", ordered) + [""]
    by_ticker = dict(sorted(card.by_ticker.items()))
    lines += ["## By ticker", ""] + _table("Ticker", by_ticker) + [""]
    lines.append(f"Pending: {card.pending} · Unscored: {card.skipped}")
    return "\n".join(lines) + "\n"
