# CLAUDE.md — Indian Stock Market Multi-Agent Trading System

## Project Overview
Multi-agent AI trading system for NSE/BSE using CrewAI + Gemini 2.0 Flash.
This is a financial system — correctness and safety are paramount.

## Architecture Decisions (DO NOT CHANGE WITHOUT DISCUSSION)
- All agents extend src/agents/base_agent.py (LLM config, retry logic)
- All tools extend src/tools/base_tool.py (rate limiting, error handling)
- Pydantic models in src/models/ are the single source of truth for data contracts
- Broker interactions go through the abstract BaseBroker interface only
- Redis is used for rate limit counters and intra-run state; PostgreSQL for persistence

## LLM Configuration
- PRODUCTION: Google Gemini 2.0 Flash (model: gemini-2.0-flash-exp)
- LOCAL DEV: Ollama with llama3.1:8b or phi3:medium (set LLM_PROVIDER=ollama)
- LLM routing is in src/config/llm_config.py — always use get_llm() factory function
- Never instantiate ChatGoogleGenerativeAI or ChatOllama directly in agents

## Indian Market Constraints (CRITICAL)
- Market hours: 9:15 AM to 3:30 PM IST Monday-Friday (excluding NSE holidays)
- Pre-market session: 9:00-9:15 AM IST (order entry only, no execution)
- Circuit breakers: Stocks have 5%, 10%, or 20% daily price bands — check before ordering
- NSE tick size: ₹0.05 for most equities; price validation must use round_to_tick()
- F&O lot sizes: Must use NIFTY_LOTS dict from src/config/market_hours.py
- T+1 settlement for equities; positions must account for unsettled funds
- BSE scrip code is numeric; NSE symbol is alphanumeric — never mix them

## Broker API Notes
- Kite Connect access token expires daily — refresh via scripts/setup_kite.py
- Never hardcode access tokens; always load from environment: KITE_ACCESS_TOKEN
- Kite API rate limits: 3 req/sec for order APIs, 10 req/sec for data APIs
- Paper trading mode: set BROKER_MODE=paper to use PaperBroker (no real orders)

## Adding New Agents
1. Create src/agents/your_agent.py extending BaseAgent
2. Define tools in src/tools/ extending BaseTool
3. Add Pydantic output model in src/models/
4. Wire into appropriate workflow (buyer/seller) in src/workflows/
5. Add unit tests in tests/unit/test_your_agent.py

## Adding New Tools
1. Extend BaseTool from src/tools/base_tool.py
2. Implement _run() method (sync); use @retry() decorator for network calls
3. Declare rate_limit_key and rate_limit_per_minute in class body
4. Use httpx.AsyncClient for HTTP — never requests library
5. Cache responses in Redis with TTL: market data=60s, fundamentals=3600s

## NSE Scraping Pattern
NSE website requires session cookies. Use NSESession class from src/tools/market/nse_fetcher.py.
Always set headers: {"User-Agent": "...", "Referer": "https://www.nseindia.com"}
NSE returns 401 after ~5 min of inactivity — the NSESession auto-refreshes.

## Testing Strategy
- Unit tests use recorded fixtures (tests/fixtures/) — never make live API calls in tests
- Use pytest-asyncio for async tool tests
- Mock Kite Connect with tests/fixtures/mock_kite.py
- Run: make test (unit), make test-integration (needs Redis)

## Windows-Specific Notes (IMPORTANT)
- Shell: PowerShell 7 (pwsh). All terminal commands use PS> prompt.
- Path separators: Use forward slashes / in Python code (os.path.join handles this).
- Virtual environment activation: .venv\Scripts\Activate.ps1 (not source .venv/bin/activate)
- Python binary: "python" (not "python3") on Windows
- Line endings: All files use LF. Git configured with core.autocrlf=false.
- Environment variables: Loaded from .env.local via python-dotenv (same as Linux).
- Docker: Uses WSL2 backend. All docker compose commands work identically.
- Make: Available via Scoop. Makefile targets work natively.

## Code Style
- Python 3.11+ type hints everywhere (use | instead of Union)
- Docstrings on all public classes and methods
- structlog for logging — never use print() in production code
- All monetary values in float INR (Indian Rupees)
- All timestamps as datetime with IST timezone (pytz.timezone("Asia/Kolkata"))

## Environment Variables (see .env.example)
GEMINI_API_KEY, KITE_API_KEY, KITE_ACCESS_TOKEN, KITE_SECRET
UPSTOX_API_KEY, UPSTOX_SECRET, TAVILY_API_KEY, GNEWS_API_KEY
REDIS_URL, DATABASE_URL, LLM_PROVIDER (gemini|ollama), BROKER_MODE (live|paper)

## SEBI Compliance
- This tool produces RECOMMENDATIONS only. Order execution requires explicit --execute flag.
- All recommendations are logged to database with timestamp, rationale, and agent versions.
- Never store KITE credentials in code, logs, or database — environment only.
- System does not constitute SEBI-registered investment advice. Add disclaimer to all reports.
