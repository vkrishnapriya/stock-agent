"""
src/agents/market_scanner.py
MarketScannerAgent — algorithmic screener for NSE stock universes.

Downloads 3 months of daily OHLCV data in a single yfinance batch call,
then ranks stocks by 1-month momentum and relative volume.

This is a deterministic data-pipeline step (not an LLM agent) — it runs
quickly without consuming LLM rate limits and feeds candidates into the
NewsSentimentAgent and the deep TA/RM/FA/CA analysis crews.

Usage::

    scanner = MarketScannerAgent()
    results = scanner.scan("NIFTY50", top_n=20)
    for entry in results.entries:
        print(entry.rank, entry.symbol, entry.momentum_1m_pct)
"""

from __future__ import annotations

from typing import Any

import structlog

from src.models.scan import ScanEntry, ScanResults

log = structlog.get_logger(__name__)

# Approximate trading-day windows
_DAYS_1M = 21
_DAYS_20D = 20

# ---------------------------------------------------------------------------
# NSE universe symbol lists (Phase 1)
# ---------------------------------------------------------------------------

NIFTY50_SYMBOLS: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "DMART", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TECHM", "HCLTECH", "POWERGRID", "NTPC",
    "ONGC", "COALINDIA", "INDUSINDBK", "BAJAJFINSV", "TATASTEEL",
    "TATAMOTORS", "M&M", "DRREDDY", "CIPLA", "DIVISLAB",
    "EICHERMOT", "BAJAJ-AUTO", "HEROMOTOCO", "SBILIFE", "HDFCLIFE",
    "APOLLOHOSP", "BPCL", "GRASIM", "BRITANNIA", "TATACONSUM",
    "ADANIPORTS", "ADANIENT", "JSWSTEEL", "HINDALCO", "UPL",
]

NIFTY100_SYMBOLS: list[str] = NIFTY50_SYMBOLS + [
    "PIDILITIND", "BERGEPAINT", "HAVELLS", "DABUR", "MARICO",
    "GODREJCP", "COLPAL", "MCDOWELL-N", "PAGEIND", "VOLTAS",
    "AMBUJACEM", "SHREECEM", "RAMCOCEM", "ACC", "DALBHARAT",
    "BANKBARODA", "CANBK", "PNB", "UNIONBANK", "IDFCFIRSTB",
    "BANDHANBNK", "FEDERALBNK", "RBLBANK", "AUBANK", "CSBBANK",
    "MFSL", "ICICIPRULI", "ICICIGI", "NIACL", "GICRE",
    "RECLTD", "PFC", "IRFC", "NHPC", "SJVN",
    "NMDC", "VEDL", "NATIONALUM", "SAIL", "JINDALSTEL",
    "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY",
    "CHOLAFIN", "MUTHOOTFIN", "BAJAJHLDNG", "LICHSGFIN", "SHRIRAMFIN",
]

_UNIVERSES: dict[str, list[str]] = {
    "NIFTY50": NIFTY50_SYMBOLS,
    "NIFTY100": NIFTY100_SYMBOLS,
    # NIFTY500 falls back to NIFTY100 in Phase 1 (full 500-symbol download
    # is too slow for real-time use; Phase 2 will use NSE bhavcopy instead).
    "NIFTY500": NIFTY100_SYMBOLS,
}

_SECTOR_MAP: dict[str, str] = {
    "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking",
    "INFY": "IT", "ICICIBANK": "Banking", "HINDUNILVR": "FMCG",
    "ITC": "FMCG", "SBIN": "Banking", "BHARTIARTL": "Telecom",
    "KOTAKBANK": "Banking", "LT": "Capital Goods", "AXISBANK": "Banking",
    "ASIANPAINT": "Paints", "DMART": "Retail", "MARUTI": "Auto",
    "SUNPHARMA": "Pharma", "TITAN": "Consumer", "BAJFINANCE": "NBFC",
    "WIPRO": "IT", "ULTRACEMCO": "Cement", "NESTLEIND": "FMCG",
    "TECHM": "IT", "HCLTECH": "IT", "POWERGRID": "Power",
    "NTPC": "Power", "ONGC": "Oil & Gas", "COALINDIA": "Mining",
    "INDUSINDBK": "Banking", "BAJAJFINSV": "NBFC", "TATASTEEL": "Steel",
    "TATAMOTORS": "Auto", "M&M": "Auto", "DRREDDY": "Pharma",
    "CIPLA": "Pharma", "DIVISLAB": "Pharma", "EICHERMOT": "Auto",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "SBILIFE": "Insurance",
    "HDFCLIFE": "Insurance", "APOLLOHOSP": "Healthcare", "BPCL": "Oil & Gas",
    "GRASIM": "Diversified", "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "ADANIPORTS": "Infrastructure", "ADANIENT": "Diversified",
    "JSWSTEEL": "Steel", "HINDALCO": "Metals", "UPL": "Agri",
    "ZOMATO": "Consumer Tech", "NYKAA": "Consumer Tech",
    # NIFTY100 additions
    "PIDILITIND": "Chemicals",    "BERGEPAINT": "Paints",      "HAVELLS": "Consumer Electricals",
    "DABUR": "FMCG",              "MARICO": "FMCG",            "GODREJCP": "FMCG",
    "COLPAL": "FMCG",             "MCDOWELL-N": "Beverages",   "PAGEIND": "Textiles",
    "VOLTAS": "Consumer Electricals",
    "AMBUJACEM": "Cement",        "SHREECEM": "Cement",        "RAMCOCEM": "Cement",
    "ACC": "Cement",              "DALBHARAT": "Cement",
    "BANKBARODA": "Banking",      "CANBK": "Banking",          "PNB": "Banking",
    "UNIONBANK": "Banking",       "IDFCFIRSTB": "Banking",     "BANDHANBNK": "Banking",
    "FEDERALBNK": "Banking",      "RBLBANK": "Banking",        "AUBANK": "Banking",
    "CSBBANK": "Banking",
    "MFSL": "Insurance",          "ICICIPRULI": "Insurance",   "ICICIGI": "Insurance",
    "NIACL": "Insurance",         "GICRE": "Insurance",
    "RECLTD": "Power Finance",    "PFC": "Power Finance",      "IRFC": "Infra Finance",
    "NHPC": "Power",              "SJVN": "Power",
    "NMDC": "Mining",             "VEDL": "Metals",            "NATIONALUM": "Metals",
    "SAIL": "Steel",              "JINDALSTEL": "Steel",
    "PAYTM": "Fintech",           "POLICYBZR": "Insurtech",    "DELHIVERY": "Logistics",
    "CHOLAFIN": "NBFC",           "MUTHOOTFIN": "NBFC",        "BAJAJHLDNG": "NBFC",
    "LICHSGFIN": "NBFC",          "SHRIRAMFIN": "NBFC",
}


def _get_universe(name: str) -> list[str]:
    key = name.strip().upper()
    if key not in _UNIVERSES:
        raise ValueError(
            f"Unknown universe '{name}'. Valid options: {sorted(_UNIVERSES)}"
        )
    return _UNIVERSES[key]


# ---------------------------------------------------------------------------
# MarketScannerAgent
# ---------------------------------------------------------------------------


class MarketScannerAgent:
    """Algorithmic screener that ranks NSE stocks by 1-month momentum.

    Downloads price + volume data via yfinance, computes 1M return and a
    20-day volume ratio, then returns the top-*n* symbols as
    :class:`~src.models.scan.ScanResults`.

    Example::

        scanner = MarketScannerAgent()
        results = scanner.scan("NIFTY50", top_n=15)
        print(results.shortlist(5))
    """

    def scan(self, universe: str = "NIFTY500", top_n: int = 20) -> ScanResults:
        """Screen *universe* and return top-*top_n* stocks by 1M momentum.

        Args:
            universe: ``"NIFTY50"``, ``"NIFTY100"``, or ``"NIFTY500"``.
            top_n:    Maximum entries to include in the result.

        Returns:
            :class:`~src.models.scan.ScanResults` sorted by descending
            1-month momentum, each entry ranked from 1 (best).
        """
        import yfinance as yf

        symbols = _get_universe(universe)
        log.info("market_scanner.start", universe=universe, total=len(symbols))

        ns_symbols = [f"{s}.NS" for s in symbols]
        df = yf.download(
            ns_symbols,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        entries: list[ScanEntry] = []
        for sym, ns_sym in zip(symbols, ns_symbols):
            try:
                close, vol = self._extract_series(df, ns_sym, len(ns_symbols))
                if close is None or close.empty or len(close) < 5:
                    continue

                current_price = float(close.iloc[-1])

                # 1-month momentum
                if len(close) >= _DAYS_1M:
                    momentum_1m = (current_price / float(close.iloc[-_DAYS_1M]) - 1) * 100
                else:
                    momentum_1m = (current_price / float(close.iloc[0]) - 1) * 100

                # Volume ratio: today vs 20-day average
                vol_ratio = 1.0
                if vol is not None and not vol.empty and len(vol) >= _DAYS_20D + 1:
                    avg_vol = float(vol.iloc[-_DAYS_20D:-1].mean())
                    if avg_vol > 0:
                        vol_ratio = float(vol.iloc[-1]) / avg_vol

                entries.append(
                    ScanEntry(
                        symbol=sym,
                        sector=_SECTOR_MAP.get(sym, ""),
                        momentum_1m_pct=round(momentum_1m, 2),
                        volume_ratio=round(vol_ratio, 2),
                        last_price=round(current_price, 2),
                        rank=1,  # reassigned below
                    )
                )
            except Exception as exc:
                log.warning("market_scanner.symbol_failed", symbol=sym, error=str(exc))

        # Sort by momentum, assign final ranks, cap at top_n
        entries.sort(key=lambda e: e.momentum_1m_pct, reverse=True)
        ranked = [e.model_copy(update={"rank": i + 1}) for i, e in enumerate(entries)]
        top = ranked[:top_n]

        log.info(
            "market_scanner.done",
            universe=universe,
            scanned=len(entries),
            returned=len(top),
        )
        return ScanResults(universe=universe, entries=top, total_scanned=len(symbols))

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_series(df: Any, ns_sym: str, total: int) -> tuple[Any, Any]:
        """Extract (Close, Volume) Series from a flat or MultiIndex DataFrame."""
        import pandas as pd

        try:
            close = df["Close"] if total == 1 else df["Close"][ns_sym]
            close = close.dropna()
        except (KeyError, TypeError):
            return pd.Series(dtype=float), None

        try:
            vol = df["Volume"] if total == 1 else df["Volume"][ns_sym]
            vol = vol.dropna()
        except (KeyError, TypeError):
            vol = None

        return close, vol
