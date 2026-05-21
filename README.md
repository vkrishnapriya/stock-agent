# stock-agent

Indian Stock Market Multi-Agent AI Trading System built with [CrewAI](https://www.crewai.com/) and Claude (Anthropic). Analyses NSE/BSE equities and produces actionable **BUY / SELL / HOLD** recommendations with entry zones, stop-losses, price targets, and Reward:Risk ratios.

> **Disclaimer:** This tool produces research recommendations only. It does not constitute SEBI-registered investment advice. All order execution requires explicit user action. Use at your own risk.

---

## Features

- **Buy workflow** — scans a stock universe (NIFTY50 / NIFTY100 / NIFTY500 / NIFTYMIDCAP150), filters by momentum, sentiment, technicals, fundamentals, and risk, then ranks buy candidates
- **Sell workflow** — analyses an existing portfolio and recommends SELL or HOLD for each holding
- **Market scan** — quick momentum scan showing top movers by 1-month return
- **Three-mode entry logic** — BREAKOUT, PULLBACK, or CURRENT_PRICE entries anchored to current market price with minimum 2:1 Reward:Risk enforcement
- **Batch LLM calls** — all stocks analysed in a single LLM prompt per agent (not one call per stock)
- **Parallel data fetching** — concurrent API calls for price data, news, whale activity, and fundamentals
- **Paper trading safe** — `BROKER_MODE=paper` prevents any real orders

---

## Architecture

```
CLI (src/cli.py)
  └── BuyerWorkflow / SellerWorkflow  (src/workflows/)
        ├── MarketScannerAgent         — momentum screen, universe filtering
        ├── NewsSentimentAgent         — news + whale activity, batch LLM
        ├── TechnicalAnalysisAgent     — RSI, MACD, support/resistance, volume
        ├── FinancialAnalystAgent      — P/E, EPS, revenue growth, fundamentals
        ├── CompetitorAnalystAgent     — sector peer comparison
        └── RiskManagerAgent           — position sizing, stop-loss, circuit filters
```

Each agent extends `src/agents/base_agent.py`. Each tool extends `src/tools/base_tool.py` (rate limiting, retry, Redis caching).

LLM is configured via `src/config/llm_config.get_llm()` — never instantiate providers directly.

---

## Requirements

- Python 3.11+
- Redis (for rate limiting and caching)
- PostgreSQL (for persistence, optional for recommendations-only use)
- API keys (see Environment Variables below)

---

## Installation

```bash
# Clone
git clone <repo-url>
cd stock-agent

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development (linting, testing)
pip install -r requirements-dev.txt
```

---

## Environment Variables

Copy `.env.example` to `.env.local` and fill in the values:

```bash
# LLM
ANTHROPIC_API_KEY=sk-ant-...          # Primary LLM (Claude)
GEMINI_API_KEY=...                    # Alternative LLM (Gemini)

# LLM provider selection: claude | gemini | groq | ollama
LLM_PROVIDER=claude

# Broker
KITE_API_KEY=...
KITE_ACCESS_TOKEN=...                 # Refresh daily via scripts/setup_kite.py
KITE_SECRET=...
BROKER_MODE=paper                     # paper | live

# News / Web
TAVILY_API_KEY=...
GNEWS_API_KEY=...

# Persistence
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/stockagent
```

For local development without a real broker, set `BROKER_MODE=paper`.

---

## CLI Usage

### Buy — find new stocks to buy

```bash
# Find IT sector opportunities (default: NIFTY500, Rs 1,00,000 cash)
python -m src.cli buy "IT sector opportunities under Rs 2000"

# Specify universe and capital
python -m src.cli buy "momentum stocks with strong earnings" --universe NIFTY100 --cash 500000

# Midcap screen
python -m src.cli buy "midcap growth stocks" --universe NIFTYMIDCAP150 --cash 300000

# Infrastructure stocks
python -m src.cli buy "infrastructure and capital goods stocks for short term" --universe NIFTY500
```

**Available universes:** `NIFTY50`, `NIFTY100`, `NIFTY500`, `NIFTYMIDCAP150`, `NIFTYMIDCAP50`

**Output columns:**

| Column | Description |
|---|---|
| Rank | Composite score rank |
| Symbol | NSE ticker |
| Entry Type | BREAKOUT / PULLBACK / CURRENT_PRICE |
| Score | 0–100 composite buy score |
| Entry Zone Rs | Executable price range to place the buy order |
| Stop Rs | Stop-loss price (max 8% below entry) |
| Target Rs | Price target (minimum 2:1 Reward:Risk enforced) |
| Reward:Risk | (Target − Entry) / (Entry − Stop) — minimum 1.5:1 to qualify |
| Allocation Rs | Suggested capital to deploy |

### Sell — analyse an existing portfolio

```bash
# Default portfolio path
python -m src.cli sell

# Custom portfolio file
python -m src.cli sell --portfolio data/portfolios/my_portfolio.json
```

Portfolio format (JSON):

```json
{
  "portfolio_id": "my-portfolio",
  "holdings": [
    {
      "symbol": "RELIANCE",
      "exchange": "NSE",
      "quantity": 10,
      "avg_buy_price": 2450.75,
      "sector": "Energy"
    }
  ],
  "cash_balance_inr": 50000.00,
  "risk_profile": "moderate"
}
```

See `data/portfolios/sample_portfolio.json` for a full example.

### Scan — quick momentum screen

```bash
# Top 20 movers in NIFTY500 (default)
python -m src.cli scan

# Top 10 in NIFTY50
python -m src.cli scan NIFTY50 --top 10

# NIFTYMIDCAP150
python -m src.cli scan NIFTYMIDCAP150 --top 30
```

---

## Entry Point Logic

Entries are anchored to the **current market price** — not historical support levels — to ensure recommendations are executable.

**Mode selection (automatic):**

| Mode | Condition | Entry Zone |
|---|---|---|
| BREAKOUT | Volume HIGH + price >= resistance × 0.98 | `[last_price, last_price × 1.005]` |
| PULLBACK | RSI < 45 + price <= support × 1.03 | `[support, support × 1.02]` |
| CURRENT_PRICE | Everything else | `[last_price, last_price × 1.005]` |

**Stop-loss priority** (in order):
1. Risk Manager agent suggestion — if below entry and within 8%
2. Support level — if below entry
3. Fallback: entry × 0.95 (5% hard floor)

**Target:** `max(resistance_level, entry_upper + 2 × risk_per_share)` — ensures minimum 2:1 Reward:Risk.

**Discard rule:** Candidates with Reward:Risk below 1.5:1 are rejected.

See `docs/entry_point_calculation_plan.md` for full design documentation.

---

## Reports

All workflow runs save a timestamped JSON report to `data/reports/`:

```
data/reports/
  20260521_143022_buy.json
  20260521_091500_sell.json
  20260521_082200_scan.json
```

---

## Development

```bash
# Run tests
make test

# Run with coverage
make test-integration   # requires Redis

# Lint
ruff check src/
mypy src/

# Format
black src/
```

### Adding a new agent

1. Create `src/agents/your_agent.py` extending `BaseAgent`
2. Define tools in `src/tools/` extending `BaseTool`
3. Add Pydantic output model in `src/models/`
4. Wire into the appropriate workflow in `src/workflows/`
5. Add unit tests in `tests/unit/test_your_agent.py`

### Refreshing Kite access token

```bash
python scripts/setup_kite.py
```

---

## Indian Market Notes

- Market hours: 9:15 AM – 3:30 PM IST, Monday–Friday (NSE holidays excluded)
- All monetary values are in INR (Indian Rupees)
- Circuit breakers: 5%, 10%, or 20% daily price bands — checked before recommendations
- NSE tick size: Rs 0.05 — all prices validated via `round_to_tick()`
- T+1 equity settlement — unsettled funds accounted for in position sizing
- NSE HTTP sessions auto-refresh via `NSESession` (cookies expire after ~5 min)

---

## Project Structure

```
stock-agent/
  src/
    agents/          # CrewAI agent definitions
    tools/
      market/        # yfinance, NSE fetcher, peer performance
      technical/     # indicator engine, support/resistance
      news/          # Tavily, GNews, NSE announcements, whale tracker
      risk/          # position sizer, liquidity checker, circuit checker
    workflows/       # BuyerWorkflow, SellerWorkflow, Director
    models/          # Pydantic data contracts (signals, recommendations, portfolio)
    config/          # LLM config, settings, market hours
    broker/          # BaseBroker, KiteBroker, PaperBroker
    utils/           # logger, retry, rate limiter, report formatter
  tests/
    unit/
    integration/
    fixtures/        # mock_kite.py and recorded API responses
  scripts/
    setup_kite.py    # daily Kite access token refresh
    download_bhav.py # NSE bhavcopy downloader
  data/
    portfolios/      # portfolio JSON files
    reports/         # generated recommendation reports
  docs/
    entry_point_calculation_plan.md
```
