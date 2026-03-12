"""
src/utils/report_formatter.py
Convert a FinalReport to a Markdown document and save it to disk.

Usage::

    from src.utils.report_formatter import format_recommendations, save_report

    md  = format_recommendations(report)
    out = save_report(report, output_dir="reports/")
    print(f"Saved to {out}")
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytz

from src.models.recommendations import BuyCandidate, StockDecision
from src.models.reports import FinalReport

_IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_inr(value: float) -> str:
    """Format a float as Indian Rupees, e.g. ₹2,500.75"""
    return f"₹{value:,.2f}"


def _fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string, e.g. 82.0%"""
    return f"{value:.{decimals}f}%"


def _confidence_bar(confidence: float, width: int = 10) -> str:
    """Return a simple ASCII progress bar, e.g. '████████░░ 82%'"""
    filled = round(confidence * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {_fmt_pct(confidence * 100, 0)}"


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", text)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_sell_hold(decisions: list[StockDecision]) -> str:
    if not decisions:
        return ""

    sells = [d for d in decisions if d.action == "SELL"]
    holds = [d for d in decisions if d.action == "HOLD"]
    lines: list[str] = []

    for label, group in [("SELL", sells), ("HOLD", holds)]:
        if not group:
            continue
        lines.append(f"## {label} Decisions ({len(group)})\n")
        lines.append("| Symbol | Exchange | Confidence | Stop Loss | Rationale |")
        lines.append("|--------|----------|-----------|-----------|-----------|")
        for d in group:
            rationale_short = d.rationale[:80] + ("…" if len(d.rationale) > 80 else "")
            lines.append(
                f"| **{d.symbol}** | {d.exchange} "
                f"| {_confidence_bar(d.confidence)} "
                f"| {_fmt_inr(d.stop_loss_inr)} "
                f"| {rationale_short} |"
            )
        lines.append("")

        # Detail block per decision
        for d in group:
            lines.append(f"### {d.symbol} — {d.action}")
            lines.append(f"- **Exchange:** {d.exchange}")
            lines.append(f"- **Confidence:** {_fmt_pct(d.confidence * 100)}")
            lines.append(f"- **Stop Loss:** {_fmt_inr(d.stop_loss_inr)}")
            lines.append(f"- **Rationale:** {d.rationale}")
            lines.append(f"- **Generated at:** {d.generated_at.strftime('%Y-%m-%d %H:%M %Z')}")
            lines.append("")

    return "\n".join(lines)


def _render_buy_candidates(candidates: list[BuyCandidate]) -> str:
    if not candidates:
        return ""

    lines: list[str] = [f"## Buy Candidates ({len(candidates)})\n"]
    lines.append(
        "| Symbol | Exchange | Score | Allocation | Entry Zone | Stop Loss | Target |"
    )
    lines.append(
        "|--------|----------|-------|-----------|-----------|-----------|--------|"
    )
    for b in candidates:
        lines.append(
            f"| **{b.symbol}** | {b.exchange} "
            f"| {b.score:.1f}/100 "
            f"| {_fmt_inr(b.suggested_allocation_inr)} "
            f"| {_fmt_inr(b.entry_zone.lower_inr)}–{_fmt_inr(b.entry_zone.upper_inr)} "
            f"| {_fmt_inr(b.stop_loss_inr)} "
            f"| {_fmt_inr(b.target_inr)} |"
        )
    lines.append("")

    for b in candidates:
        upside = ((b.target_inr - b.entry_zone.upper_inr) / b.entry_zone.upper_inr) * 100
        downside = ((b.entry_zone.lower_inr - b.stop_loss_inr) / b.entry_zone.lower_inr) * 100
        rr_ratio = upside / downside if downside else 0
        lines.append(f"### {b.symbol} — BUY CANDIDATE")
        lines.append(f"- **Exchange:** {b.exchange}")
        lines.append(f"- **Composite Score:** {b.score:.1f} / 100")
        lines.append(f"- **Suggested Allocation:** {_fmt_inr(b.suggested_allocation_inr)}")
        lines.append(
            f"- **Entry Zone:** {_fmt_inr(b.entry_zone.lower_inr)} – "
            f"{_fmt_inr(b.entry_zone.upper_inr)}"
        )
        lines.append(f"- **Stop Loss:** {_fmt_inr(b.stop_loss_inr)} ({_fmt_pct(downside)} risk)")
        lines.append(f"- **Target:** {_fmt_inr(b.target_inr)} ({_fmt_pct(upside)} upside)")
        lines.append(f"- **Risk/Reward:** 1 : {rr_ratio:.1f}")
        lines.append(f"- **Generated at:** {b.generated_at.strftime('%Y-%m-%d %H:%M %Z')}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_recommendations(report: FinalReport) -> str:
    """Render *report* as a Markdown string.

    The document contains:
    - Header with timestamp, workflow type, and decision count
    - Summary paragraph
    - SELL / HOLD decision table with detail blocks
    - Buy candidate table with detail blocks
    - SEBI disclaimer footer

    Args:
        report: A :class:`~src.models.reports.FinalReport` instance.

    Returns:
        A Markdown-formatted string ready for display or file output.
    """
    ts = report.timestamp.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S IST")

    stock_decisions: list[StockDecision] = [
        d for d in report.decisions if isinstance(d, StockDecision)
    ]
    buy_candidates: list[BuyCandidate] = [
        d for d in report.decisions if isinstance(d, BuyCandidate)
    ]

    sections: list[str] = []

    # --- Header ---
    sections.append(f"# Trading Report — {report.workflow_type}\n")
    sections.append(f"**Generated:** {ts}  ")
    sections.append(f"**Workflow:** {report.workflow_type}  ")
    sections.append(f"**Total Decisions:** {report.decision_count}\n")
    sections.append("---\n")

    # --- Summary ---
    sections.append("## Summary\n")
    sections.append(report.summary)
    sections.append("\n---\n")

    # --- Decisions ---
    sell_hold_md = _render_sell_hold(stock_decisions)
    if sell_hold_md:
        sections.append(sell_hold_md)
        sections.append("---\n")

    buy_md = _render_buy_candidates(buy_candidates)
    if buy_md:
        sections.append(buy_md)
        sections.append("---\n")

    # --- Disclaimer ---
    sections.append(f"> **Disclaimer:** {report.disclaimer}\n")

    return "\n".join(sections)


def save_report(report: FinalReport, output_dir: str | Path = "reports") -> Path:
    """Persist *report* to a Markdown file in *output_dir*.

    The filename is derived from the workflow type and IST timestamp::

        SELLER_20260312_143000.md

    Args:
        report:     A :class:`~src.models.reports.FinalReport` instance.
        output_dir: Directory path (created if it does not exist).

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts_str = report.timestamp.astimezone(_IST).strftime("%Y%m%d_%H%M%S")
    filename = f"{_slugify(report.workflow_type)}_{ts_str}.md"
    file_path = out / filename

    content = format_recommendations(report)
    file_path.write_text(content, encoding="utf-8")
    return file_path


__all__ = ["format_recommendations", "save_report"]
