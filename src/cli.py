"""
src/cli.py
Typer CLI for the stock-agent system.

Commands
--------
  sell  --portfolio PATH   Analyse an existing portfolio (SELL / HOLD decisions)
  buy   PROMPT             Find buy candidates from a stock universe
  scan  UNIVERSE           Quick momentum scan of a universe

All commands save a JSON report to data/reports/.

Usage::

    stock-agent sell --portfolio data/portfolios/portfolio.json
    stock-agent buy "Find IT sector opportunities under Rs 2000"
    stock-agent scan NIFTY500
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz
import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logger import configure_logging

log = structlog.get_logger(__name__)

_IST = pytz.timezone("Asia/Kolkata")
_REPORTS_DIR = Path("data/reports")

# On Windows the default cp1252 terminal can't render Unicode (Rs, braille
# spinners, etc.). Reconfigure stdout/stderr to UTF-8 before Rich starts.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

app = typer.Typer(
    name="stock-agent",
    help="Indian Stock Market Multi-Agent AI (CrewAI + Gemini 2.5 Flash)",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_reports_dir() -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_report(data: dict, prefix: str) -> Path:
    """Serialise *data* to a timestamped JSON file under data/reports/."""
    _ensure_reports_dir()
    ts = datetime.now(_IST).strftime("%Y%m%d_%H%M%S")
    path = _REPORTS_DIR / f"{ts}_{prefix}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def _load_settings() -> None:
    """Initialise logging from application settings."""
    try:
        from src.config.settings import get_settings
        configure_logging(get_settings().log_level)
    except Exception:
        configure_logging("INFO")


# ---------------------------------------------------------------------------
# sell command
# ---------------------------------------------------------------------------


@app.command()
def sell(
    portfolio: str = typer.Option(
        "data/portfolios/portfolio.json",
        "--portfolio",
        "-p",
        help="Path to portfolio JSON or CSV file.",
        show_default=True,
    ),
) -> None:
    """Analyse an existing portfolio and recommend SELL or HOLD for each holding."""
    _load_settings()

    from src.models.portfolio import Portfolio
    from src.workflows.seller_workflow import SellerWorkflow

    console.print(Panel(f"[bold cyan]Seller Workflow[/bold cyan]\nPortfolio: {portfolio}"))

    try:
        p = Portfolio.load_from_file(portfolio)
    except FileNotFoundError:
        console.print(f"[red]Portfolio file not found:[/red] {portfolio}")
        raise typer.Exit(1)

    console.print(
        f"Loaded [bold]{p.portfolio_id}[/bold] — "
        f"{len(p.holdings)} holding(s), "
        f"Rs{p.available_cash_inr:,.0f} cash"
    )

    with console.status("[yellow]Running analysis crews…[/yellow]"):
        report = SellerWorkflow().run_as_report(p)

    # Display table
    table = Table(title="Sell / Hold Decisions", show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan")
    table.add_column("Action", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Stop-Loss Rs", justify="right")
    table.add_column("Rationale")

    for d in report.sell_decisions + report.hold_decisions:
        colour = "red" if d.action == "SELL" else "green"
        table.add_row(
            d.symbol,
            f"[{colour}]{d.action}[/{colour}]",
            f"{d.confidence:.2%}",
            f"{d.stop_loss_inr:,.2f}",
            d.rationale[:80],
        )

    console.print(table)
    console.print(f"\n[dim]{report.summary}[/dim]")

    # Save report
    path = _save_report(report.model_dump(), "sell")
    console.print(f"\n[green]Report saved →[/green] {path}")


# ---------------------------------------------------------------------------
# buy command
# ---------------------------------------------------------------------------


@app.command()
def buy(
    prompt: str = typer.Argument(
        ...,
        help='Natural-language buy intent, e.g. "IT sector opportunities under Rs 2000".',
    ),
    universe: str = typer.Option(
        "NIFTY50",
        "--universe",
        "-u",
        help="Stock universe to scan: NIFTY50, NIFTY100, NIFTY500.",
        show_default=True,
    ),
    available_cash: float = typer.Option(
        100_000.0,
        "--cash",
        "-c",
        help="Available capital for new positions (INR).",
        show_default=True,
    ),
) -> None:
    """Find buy candidates matching PROMPT from the given stock universe."""
    _load_settings()

    from src.workflows.buyer_workflow import BuyerWorkflow

    console.print(
        Panel(
            f"[bold cyan]Buyer Workflow[/bold cyan]\n"
            f"Prompt: {prompt}\n"
            f"Universe: {universe}  |  Available cash: Rs{available_cash:,.0f}"
        )
    )

    with console.status("[yellow]Scanning and analysing…[/yellow]"):
        report = BuyerWorkflow().run_as_report(
            universe=universe,
            prompt=prompt,
            available_cash=available_cash,
        )

    candidates = report.buy_candidates
    if not candidates:
        console.print("[yellow]No buy candidates found after filtering.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Buy Candidates (ranked)", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right")
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Entry Zone Rs", justify="right")
    table.add_column("Stop Rs", justify="right")
    table.add_column("Target Rs", justify="right")
    table.add_column("Allocation Rs", justify="right")

    for i, c in enumerate(candidates, 1):
        table.add_row(
            str(i),
            c.symbol,
            f"{c.score:.1f}",
            f"{c.entry_zone.lower_inr:,.0f}–{c.entry_zone.upper_inr:,.0f}",
            f"{c.stop_loss_inr:,.0f}",
            f"{c.target_inr:,.0f}",
            f"{c.suggested_allocation_inr:,.0f}",
        )

    console.print(table)
    console.print(f"\n[dim]{report.summary}[/dim]")

    path = _save_report(report.model_dump(), "buy")
    console.print(f"\n[green]Report saved →[/green] {path}")


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------


@app.command()
def scan(
    universe: str = typer.Argument(
        "NIFTY500",
        help="Universe to scan: NIFTY50, NIFTY100, NIFTY500.",
    ),
    top_n: int = typer.Option(
        20,
        "--top",
        "-n",
        help="Number of top movers to display.",
        show_default=True,
    ),
) -> None:
    """Quick momentum scan — show top movers in UNIVERSE by 1-month return."""
    _load_settings()

    from src.agents.market_scanner import MarketScannerAgent

    console.print(Panel(f"[bold cyan]Market Scan[/bold cyan]\nUniverse: {universe}  |  Top: {top_n}"))

    try:
        with console.status("[yellow]Downloading price data…[/yellow]"):
            results = MarketScannerAgent().scan(universe, top_n=top_n)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    table = Table(
        title=f"Top {top_n} Movers — {results.universe}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="right")
    table.add_column("Symbol", style="cyan")
    table.add_column("Sector")
    table.add_column("1M Return %", justify="right")
    table.add_column("Vol Ratio", justify="right")
    table.add_column("Price Rs", justify="right")

    for entry in results.entries:
        colour = "green" if entry.momentum_1m_pct >= 0 else "red"
        table.add_row(
            str(entry.rank),
            entry.symbol,
            entry.sector,
            f"[{colour}]{entry.momentum_1m_pct:+.2f}%[/{colour}]",
            f"{entry.volume_ratio:.2f}x",
            f"{entry.last_price:,.2f}",
        )

    console.print(table)
    console.print(
        f"\n[dim]Scanned {results.total_scanned} symbols "
        f"as of {results.generated_at.strftime('%Y-%m-%d %H:%M IST')}[/dim]"
    )

    # Save scan as a lightweight report
    path = _save_report(results.model_dump(), "scan")
    console.print(f"\n[green]Report saved →[/green] {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
