"""
src/config/market_hours.py
NSE/BSE market session constants and F&O lot sizes.
All times are in IST (Asia/Kolkata, UTC+5:30).
"""

from __future__ import annotations

from datetime import time

import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── Session windows ────────────────────────────────────────────────────────────
PREMARKET_OPEN = time(9, 0)    # Order entry only
MARKET_OPEN    = time(9, 15)   # Regular session opens
MARKET_CLOSE   = time(15, 30)  # Regular session closes
POST_CLOSE     = time(16, 0)   # Closing price session ends

# ── Tick size ──────────────────────────────────────────────────────────────────
DEFAULT_TICK = 0.05  # ₹0.05 for most NSE equities


def round_to_tick(price: float, tick: float = DEFAULT_TICK) -> float:
    """Round price to the nearest valid tick increment."""
    return round(round(price / tick) * tick, 2)


# ── Circuit breaker bands (index-level) ───────────────────────────────────────
INDEX_CIRCUIT_BANDS = (10, 15, 20)  # % moves that trigger market-wide halt

# ── F&O lot sizes (NSE, as of Jan 2025) ───────────────────────────────────────
# Keep this dict in sync with NSE circulars. Update quarterly on expiry revision.
NIFTY_LOTS: dict[str, int] = {
    "NIFTY":      75,
    "BANKNIFTY":  30,
    "FINNIFTY":   40,
    "MIDCPNIFTY": 75,
    "SENSEX":     10,
    "BANKEX":     15,
    # Single-stock F&O (sample — extend as needed)
    "RELIANCE":   250,
    "TCS":        150,
    "INFY":       300,
    "HDFCBANK":   550,
    "ICICIBANK":  700,
    "SBIN":      1500,
    "BAJFINANCE": 125,
    "WIPRO":     1500,
    "LT":         175,
    "AXISBANK":   625,
}


def is_market_open(current_time: time) -> bool:
    """Return True if the regular trading session is active."""
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def is_premarket(current_time: time) -> bool:
    """Return True during the pre-market (order entry) window."""
    return PREMARKET_OPEN <= current_time < MARKET_OPEN
