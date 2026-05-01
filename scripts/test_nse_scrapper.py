# scripts/test_nse_scrapper.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.market.nse_fetcher import NSESession, NSEResultsTool

SYMBOL = "INFY"

# ── 1. Quote ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  NSE Quote — {SYMBOL}")
print(f"{'='*50}")

session = NSESession()
try:
    data = session.get(f"https://www.nseindia.com/api/quote-equity?symbol={SYMBOL}")
    info       = data.get("info", {})
    price_info = data.get("priceInfo", {})
    whl        = price_info.get("weekHighLow", {})
    idhl       = price_info.get("intraDayHighLow", {})

    print(f"Company     : {info.get('companyName')}")
    print(f"Symbol      : {info.get('symbol')}")
    print(f"Last Price  : ₹{price_info.get('lastPrice')}")
    print(f"Open        : ₹{price_info.get('open')}")
    print(f"Close (prev): ₹{price_info.get('previousClose')}")
    print(f"Intraday H/L: ₹{idhl.get('max')} / ₹{idhl.get('min')}")
    print(f"52w High/Low: ₹{whl.get('max')} / ₹{whl.get('min')}")
    print(f"Change      : {price_info.get('change')} ({price_info.get('pChange')}%)")
finally:
    session.close()

# ── 2. Quarterly Results ──────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  NSE Quarterly Results — {SYMBOL}")
print(f"{'='*50}")

tool = NSEResultsTool()
raw  = tool._run(symbol=SYMBOL)
quarters = json.loads(raw)

if not quarters:
    print("No consolidated quarterly data found.")
else:
    print(f"{'Period':<12} {'Revenue (Cr)':>14} {'PAT (Cr)':>12} {'EPS':>8}  {'Audited'}")
    print("-" * 60)
    for q in quarters:
        print(
            f"{q['period']:<12}"
            f"{str(q.get('revenue_cr', 'N/A')):>14}"
            f"{str(q.get('pat_cr', 'N/A')):>12}"
            f"{str(q.get('basic_eps', 'N/A')):>8}"
            f"  {q.get('audited', '')}"
        )
