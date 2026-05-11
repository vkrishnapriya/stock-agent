"""
src/tools/news/whale_tracker.py
WhaleTrackerTool — identifies top Indian super investor activity for an NSE symbol.

Data sources (in order):
  1. NSE Bulk Deals API  — trades > 0.5% of shares; filed same day
  2. NSE Block Deals API — large negotiated off-market blocks

Whale roster covers the most-tracked Indian super investors whose names
appear in NSE client-name fields or their known investment vehicles.

Returns JSON::

    {
      "symbol": "INFY",
      "whale_buys":  [{"investor": "...", "quantity": ..., "price_inr": ..., "date": "..."}],
      "whale_sells": [...],
      "net_signal":  "BULLISH" | "BEARISH" | "NEUTRAL",
      "deal_count":  int,
      "summary":     ["BUY: Dolly Khanna bought 80,000 shares @ ₹1,480 on 02-May-2026", ...]
    }
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import structlog

from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Known Indian super investors and their investment vehicles
# Keys are UPPERCASE fragments matched against NSE client-name strings.
# ---------------------------------------------------------------------------

_WHALES: dict[str, str] = {
    # ── Value / Contrarian ───────────────────────────────────────────────
    "RADHAKISHAN DAMANI":       "Radhakishan Damani",
    "BRIGHT STAR INVESTMENTS":  "Radhakishan Damani",        # holding vehicle
    "DOLLY KHANNA":             "Dolly Khanna",
    "RAJIV KHANNA":             "Dolly Khanna",               # family account
    "VIJAY KEDIA":              "Vijay Kedia (Kedia Securities)",
    "KEDIA SECURITIES":         "Vijay Kedia",
    "ASHISH KACHOLIA":          "Ashish Kacholia",
    "LUCKY SECURITIES":         "Ashish Kacholia",            # broker vehicle
    "PORINJU VELIYATH":         "Porinju Veliyath (Equity Intelligence)",
    "EQUITY INTELLIGENCE":      "Porinju Veliyath",
    "MOHNISH PABRAI":           "Mohnish Pabrai",
    "DHANDHO INVESTMENTS":      "Mohnish Pabrai",

    # ── Rare Enterprises / Jhunjhunwala legacy ──────────────────────────
    "REKHA JHUNJHUNWALA":       "Rekha Jhunjhunwala (Rare Enterprises)",
    "RARE ENTERPRISES":         "Rekha Jhunjhunwala",
    "RAKESH JHUNJHUNWALA":      "Rakesh Jhunjhunwala (Rare Enterprises)",

    # ── Fund managers ───────────────────────────────────────────────────
    "SUNIL SINGHANIA":          "Sunil Singhania (Abakkus Asset Manager)",
    "ABAKKUS ASSET":            "Sunil Singhania",
    "RAJEEV THAKKAR":           "Rajeev Thakkar (PPFAS Mutual Fund)",
    "PPFAS":                    "Rajeev Thakkar (PPFAS)",
    "SAURABH MUKHERJEA":        "Saurabh Mukherjea (Marcellus Investment)",
    "MARCELLUS INVESTMENT":     "Saurabh Mukherjea",
    "RAAMDEO AGRAWAL":          "Raamdeo Agrawal (Motilal Oswal)",
    "MOTILAL OSWAL FINANCIAL":  "Motilal Oswal Asset Management",

    # ── Veterans ────────────────────────────────────────────────────────
    "RAMESH DAMANI":            "Ramesh Damani",
    "NEMISH SHAH":              "Nemish Shah (ENAM Securities)",
    "ENAM HOLDINGS":            "Nemish Shah",
    "MADHUSUDAN KELA":          "Madhusudan Kela",
    "NIKHIL VORA":              "Nikhil Vora (Sixth Sense Ventures)",
    "SIXTH SENSE":              "Nikhil Vora",

    # ── Global funds with heavy India focus ─────────────────────────────
    "SMALLCAP WORLD FUND":      "SmallCap World Fund (American Funds)",
    "EASTSPRING INVESTMENTS":   "Eastspring Investments",
    "GOVERNMENT OF SINGAPORE":  "GIC / Temasek (Singapore)",
    "TEMASEK":                  "GIC / Temasek (Singapore)",
}

_NSE_BULK_URL  = "https://www.nseindia.com/api/bulk-deals?symbol={symbol}"
_NSE_BLOCK_URL = "https://www.nseindia.com/api/block-deals?symbol={symbol}"

# NSE field-name variants across API versions
_CLIENT_FIELDS  = ("clientName", "clientname", "client_name", "ClientName")
_ACTION_FIELDS  = ("buySell", "buysell", "buy_sell", "BuySell", "buyorsell")
_QTY_FIELDS     = ("quantity", "qty", "Quantity")
_PRICE_FIELDS   = ("tradePrice", "tradeprice", "trade_price", "price")
_DATE_FIELDS    = ("date", "tradeDate", "trade_date", "Date")


def _pick(deal: dict[str, Any], *fields: str, default: Any = "") -> Any:
    for f in fields:
        if f in deal and deal[f] not in (None, ""):
            return deal[f]
    return default


def _match_whale(client_name: str) -> str | None:
    """Return canonical display name if *client_name* contains a known whale fragment."""
    upper = client_name.upper()
    for fragment, display in _WHALES.items():
        if fragment in upper:
            return display
    return None


class WhaleTrackerTool(BaseTool):
    """Fetches NSE bulk/block deals for a symbol and surfaces top-investor activity.

    Usage::

        tool = WhaleTrackerTool()
        result = json.loads(tool._run(symbol="INFY"))
        print(result["net_signal"])   # BULLISH / BEARISH / NEUTRAL
        for line in result["summary"]:
            print(line)
    """

    name: str = "whale_tracker"
    description: str = (
        "Fetches NSE bulk deals and block deals for a stock symbol and identifies "
        "recent buy/sell activity by top Indian super investors (Dolly Khanna, "
        "Ashish Kacholia, Vijay Kedia, Rekha Jhunjhunwala, Sunil Singhania, "
        "Radhakishan Damani, PPFAS, Marcellus, etc.). "
        "Returns JSON with whale_buys, whale_sells, net_signal, deal_count, summary."
    )
    rate_limit_key: ClassVar[str] = "nse_whale"
    rate_limit_per_minute: ClassVar[int] = 30

    def _run(self, symbol: str) -> str:  # noqa: D102
        from src.tools.market.nse_fetcher import NSESession

        symbol_upper = symbol.strip().upper()
        whale_buys:  list[dict[str, Any]] = []
        whale_sells: list[dict[str, Any]] = []

        with NSESession() as session:
            for url_template in [_NSE_BULK_URL, _NSE_BLOCK_URL]:
                url = url_template.format(symbol=symbol_upper)
                try:
                    raw = session.get(url)
                    # NSE wraps data in {"data": [...]} or returns a list directly
                    deals: list[dict[str, Any]] = (
                        raw if isinstance(raw, list) else raw.get("data", [])
                    )
                    for deal in deals:
                        client   = str(_pick(deal, *_CLIENT_FIELDS))
                        action   = str(_pick(deal, *_ACTION_FIELDS)).upper().strip()
                        qty      = _pick(deal, *_QTY_FIELDS,   default=0)
                        price    = _pick(deal, *_PRICE_FIELDS,  default=0)
                        date_str = str(_pick(deal, *_DATE_FIELDS))

                        whale = _match_whale(client)
                        if not whale:
                            continue

                        entry: dict[str, Any] = {
                            "investor":   whale,
                            "client_name": client,
                            "quantity":   qty,
                            "price_inr":  price,
                            "date":       date_str,
                        }
                        if action in ("BUY", "B", "P", "PURCHASE"):
                            whale_buys.append(entry)
                        elif action in ("SELL", "S"):
                            whale_sells.append(entry)

                except Exception as exc:
                    log.warning(
                        "whale_tracker.api_failed",
                        symbol=symbol_upper,
                        url=url,
                        error=str(exc),
                    )

        buy_count  = len(whale_buys)
        sell_count = len(whale_sells)

        if buy_count > sell_count:
            net_signal = "BULLISH"
        elif sell_count > buy_count:
            net_signal = "BEARISH"
        else:
            net_signal = "NEUTRAL"

        summary: list[str] = []
        for b in whale_buys:
            summary.append(
                f"BUY : {b['investor']} bought {int(b['quantity']):,} shares"
                f" @ \u20b9{b['price_inr']} on {b['date']}"
            )
        for s in whale_sells:
            summary.append(
                f"SELL: {s['investor']} sold {int(s['quantity']):,} shares"
                f" @ \u20b9{s['price_inr']} on {s['date']}"
            )
        if not summary:
            summary.append(
                f"No bulk/block deal activity by known super investors found for {symbol_upper}."
            )

        log.info(
            "whale_tracker.done",
            symbol=symbol_upper,
            buys=buy_count,
            sells=sell_count,
            signal=net_signal,
        )
        return json.dumps(
            {
                "symbol":      symbol_upper,
                "whale_buys":  whale_buys,
                "whale_sells": whale_sells,
                "net_signal":  net_signal,
                "deal_count":  buy_count + sell_count,
                "summary":     summary,
            },
            default=str,
        )
