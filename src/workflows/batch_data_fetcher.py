"""
src/workflows/batch_data_fetcher.py
Pre-fetches all raw tool data for a list of stocks in parallel Python threads.
No LLM calls are made here — pure data fetching and formatting.

Returns four markdown tables (one per analysis domain) ready to be embedded
directly into batch agent task descriptions.
"""

from __future__ import annotations

import json
import concurrent.futures
from typing import Any

import structlog

from src.models.portfolio import Holding
from src.models.scan import ScanEntry

log = structlog.get_logger(__name__)

_WORKERS = 5
_DEFAULT_STOP_PCT = 0.05


# ---------------------------------------------------------------------------
# Per-row fetch helpers
# ---------------------------------------------------------------------------


def _fetch_technical_row(entry: ScanEntry) -> dict[str, Any]:
    symbol = entry.symbol
    try:
        from src.tools.market.yfinance_tools import OHLCVFetchTool
        from src.tools.technical.indicator_engine import IndicatorEngineTool

        ohlcv_json = OHLCVFetchTool()._run(symbol=f"{symbol}.NS", period="3mo", interval="1d")
        sig = json.loads(IndicatorEngineTool()._run(ohlcv_json=ohlcv_json, symbol=symbol))
        return {
            "symbol": symbol,
            "price": entry.last_price,
            "rsi": sig.get("rsi"),
            "macd_hist": sig.get("macd_signal"),
            "score": sig.get("score"),
            "trend": sig.get("trend"),
            "support": sig.get("support_level_inr"),
            "resistance": sig.get("resistance_level_inr"),
            "vol_signal": sig.get("volume_signal"),
            "mom_1m_pct": entry.momentum_1m_pct,
            "vol_ratio": entry.volume_ratio,
        }
    except Exception as exc:
        log.warning("batch_fetcher.technical_failed", symbol=symbol, error=str(exc))
        return {
            "symbol": symbol, "price": entry.last_price,
            "rsi": "N/A", "macd_hist": "N/A", "score": "N/A", "trend": "N/A",
            "support": "N/A", "resistance": "N/A", "vol_signal": "N/A",
            "mom_1m_pct": entry.momentum_1m_pct, "vol_ratio": entry.volume_ratio,
        }


def _fetch_fundamental_row(entry: ScanEntry) -> dict[str, Any]:
    symbol = entry.symbol
    try:
        from src.tools.market.screener_tools import ScreenerFinancialsTool
        from src.tools.market.nse_fetcher import NSEResultsTool

        scr = json.loads(ScreenerFinancialsTool()._run(symbol=symbol))
        ratios = scr.get("key_ratios", {})
        shareholding = scr.get("shareholding_latest", {})

        # 3-year revenue CAGR from annual P&L
        sales_raw = scr.get("annual_pl", {}).get("Sales", {})
        sales_vals: list[float] = []
        for v in sales_raw.values():
            try:
                sales_vals.append(float(str(v).replace(",", "")))
            except (ValueError, TypeError):
                pass
        rev_cagr = None
        if len(sales_vals) >= 3:
            rev_cagr = round(((sales_vals[-1] / sales_vals[0]) ** (1 / (len(sales_vals) - 1)) - 1) * 100, 1)

        # EPS growth: latest quarter vs oldest in last 4
        quarters = json.loads(NSEResultsTool()._run(symbol=symbol))
        eps_growth = None
        if len(quarters) >= 2:
            eps_new = quarters[0].get("basic_eps")
            eps_old = quarters[-1].get("basic_eps")
            if eps_new is not None and eps_old and abs(eps_old) > 0:
                eps_growth = round((eps_new - eps_old) / abs(eps_old) * 100, 1)

        promoter = (
            shareholding.get("Promoters")
            or shareholding.get("Promoter")
            or "N/A"
        )
        return {
            "symbol": symbol,
            "sector": entry.sector or "N/A",
            "pe": ratios.get("Stock P/E", "N/A"),
            "pb": ratios.get("Price to book value", "N/A"),
            "roe_pct": ratios.get("Return on equity", "N/A"),
            "roce_pct": ratios.get("ROCE", "N/A"),
            "d_e": ratios.get("Debt to equity", "N/A"),
            "rev_cagr_3y": rev_cagr if rev_cagr is not None else "N/A",
            "eps_growth_pct": eps_growth if eps_growth is not None else "N/A",
            "promoter_pct": promoter,
        }
    except Exception as exc:
        log.warning("batch_fetcher.fundamental_failed", symbol=symbol, error=str(exc))
        return {
            "symbol": symbol, "sector": entry.sector or "N/A",
            "pe": "N/A", "pb": "N/A", "roe_pct": "N/A", "roce_pct": "N/A",
            "d_e": "N/A", "rev_cagr_3y": "N/A", "eps_growth_pct": "N/A",
            "promoter_pct": "N/A",
        }


def _fetch_risk_row(symbol: str, entry_price: float, portfolio_value: float) -> dict[str, Any]:
    stop_loss = round(entry_price * (1.0 - _DEFAULT_STOP_PCT), 2)
    try:
        from src.tools.risk.position_sizer import PositionSizerTool
        from src.tools.risk.circuit_checker import CircuitBreakerTool
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        sizer = json.loads(PositionSizerTool()._run(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            portfolio_value=portfolio_value,
        ))
        circuit = json.loads(CircuitBreakerTool()._run(symbol=symbol))

        adv_cr = "N/A"
        try:
            trade_val = sizer.get("position_size_inr") or (entry_price * 100)
            liq = json.loads(LiquidityCheckerTool()._run(
                symbol=f"{symbol}.NS",
                intended_trade_value_inr=float(trade_val),
            ))
            adv_cr = liq.get("adv_inr_cr", "N/A")
        except Exception:
            pass

        return {
            "symbol": symbol,
            "entry": entry_price,
            "stop": stop_loss,
            "vol_pct": sizer.get("volatility_pct", "N/A"),
            "beta": sizer.get("beta", "N/A"),
            "max_dd_pct": sizer.get("max_drawdown_pct", "N/A"),
            "circuit_band": circuit.get("band", "N/A"),
            "near_circuit": circuit.get("near_circuit", False),
            "kelly_pct": round(sizer.get("kelly_fraction", 0) * 100, 2),
            "pos_size_pct": sizer.get("position_size_pct", "N/A"),
            "adv_cr": adv_cr,
        }
    except Exception as exc:
        log.warning("batch_fetcher.risk_failed", symbol=symbol, error=str(exc))
        return {
            "symbol": symbol, "entry": entry_price, "stop": stop_loss,
            "vol_pct": "N/A", "beta": "N/A", "max_dd_pct": "N/A",
            "circuit_band": "N/A", "near_circuit": "N/A",
            "kelly_pct": "N/A", "pos_size_pct": "N/A", "adv_cr": "N/A",
        }


def _fetch_competitive_row(symbol: str, sector: str, last_price: float, momentum_1m: float) -> dict[str, Any]:
    try:
        from src.tools.market.screener_tools import CompetitorMapTool
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        peers_raw = json.loads(CompetitorMapTool()._run(symbol=symbol))
        peer_syms = [p["symbol"] for p in peers_raw[:4] if p.get("symbol") != symbol]

        symbols_csv = ",".join([symbol] + peer_syms)
        perf_data: dict[str, Any] = json.loads(PeerPricePerformanceTool()._run(symbols_csv=symbols_csv))

        target_perf = perf_data.get(symbol, {})
        ret_1m = target_perf.get("return_1m_pct", "N/A")
        ret_3m = target_perf.get("return_3m_pct", "N/A")

        # Rank among peers by 3M return
        all_3m = [(s, d.get("return_3m_pct") or 0) for s, d in perf_data.items()]
        all_3m.sort(key=lambda x: x[1], reverse=True)
        rank_3m = next((i + 1 for i, (s, _) in enumerate(all_3m) if s == symbol), "N/A")

        # Market cap rank
        target_mcap = next((p.get("market_cap") or 0 for p in peers_raw if p["symbol"] == symbol), 0)
        mcap_rank = sum(1 for p in peers_raw if (p.get("market_cap") or 0) > target_mcap) + 1

        return {
            "symbol": symbol,
            "sector": sector or "N/A",
            "peers": ",".join(peer_syms[:3]) or "N/A",
            "peer_count": len(peer_syms),
            "1m_ret_pct": ret_1m,
            "3m_ret_pct": ret_3m,
            "rank_3m": rank_3m,
            "mcap_rank": mcap_rank,
            "mom_1m_pct": momentum_1m,
        }
    except Exception as exc:
        log.warning("batch_fetcher.competitive_failed", symbol=symbol, error=str(exc))
        return {
            "symbol": symbol, "sector": sector or "N/A",
            "peers": "N/A", "peer_count": 0,
            "1m_ret_pct": "N/A", "3m_ret_pct": "N/A",
            "rank_3m": "N/A", "mcap_rank": "N/A",
            "mom_1m_pct": momentum_1m,
        }


# ---------------------------------------------------------------------------
# Markdown table formatter
# ---------------------------------------------------------------------------


def _to_md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [str(row.get(c, "N/A")) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BatchDataFetcher
# ---------------------------------------------------------------------------


class BatchDataFetcher:
    """Pre-fetches raw data for a list of stocks and returns formatted markdown tables.

    All tool calls are made from Python threads — no LLM is involved.
    The four tables returned are ready to embed directly in agent task descriptions.

    Usage::

        fetcher = BatchDataFetcher()
        tech_t, fund_t, risk_t, comp_t = fetcher.fetch_from_entries(entries, 500_000)
    """

    def __init__(self, workers: int = _WORKERS) -> None:
        self._workers = workers

    def fetch_from_entries(
        self,
        entries: list[ScanEntry],
        portfolio_value: float,
    ) -> tuple[str, str, str, str]:
        """Fetch and format tables from scanner ScanEntry objects (buyer workflow).

        Returns:
            (technical_table, fundamental_table, risk_table, competitive_table)
        """
        return self._fetch(
            symbols=[e.symbol for e in entries],
            prices={e.symbol: e.last_price for e in entries},
            sectors={e.symbol: e.sector for e in entries},
            momentum={e.symbol: e.momentum_1m_pct for e in entries},
            entries_map={e.symbol: e for e in entries},
            portfolio_value=portfolio_value,
        )

    def fetch_from_holdings(
        self,
        holdings: list[Holding],
        portfolio_value: float,
    ) -> tuple[str, str, str, str]:
        """Fetch and format tables from portfolio Holding objects (seller workflow).

        Returns:
            (technical_table, fundamental_table, risk_table, competitive_table)
        """
        return self._fetch(
            symbols=[h.symbol for h in holdings],
            prices={h.symbol: h.avg_buy_price for h in holdings},
            sectors={h.symbol: h.sector for h in holdings},
            momentum={h.symbol: 0.0 for h in holdings},
            entries_map=None,
            portfolio_value=portfolio_value,
        )

    def _fetch(
        self,
        symbols: list[str],
        prices: dict[str, float],
        sectors: dict[str, str],
        momentum: dict[str, float],
        entries_map: dict[str, ScanEntry] | None,
        portfolio_value: float,
    ) -> tuple[str, str, str, str]:
        log.info("batch_fetcher.start", stocks=len(symbols))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._workers) as pool:
            if entries_map:
                tech_futs = [pool.submit(_fetch_technical_row, entries_map[s]) for s in symbols]
                fund_futs = [pool.submit(_fetch_fundamental_row, entries_map[s]) for s in symbols]
            else:
                # For holdings: build minimal ScanEntry-like objects
                tech_futs = [
                    pool.submit(
                        _fetch_technical_row,
                        ScanEntry(
                            symbol=s, sector=sectors[s], momentum_1m_pct=0.0,
                            volume_ratio=1.0, last_price=prices[s], rank=1,
                        ),
                    )
                    for s in symbols
                ]
                fund_futs = [
                    pool.submit(
                        _fetch_fundamental_row,
                        ScanEntry(
                            symbol=s, sector=sectors[s], momentum_1m_pct=0.0,
                            volume_ratio=1.0, last_price=prices[s], rank=1,
                        ),
                    )
                    for s in symbols
                ]

            risk_futs = [
                pool.submit(_fetch_risk_row, s, prices[s], portfolio_value)
                for s in symbols
            ]
            comp_futs = [
                pool.submit(_fetch_competitive_row, s, sectors[s], prices[s], momentum[s])
                for s in symbols
            ]

        tech_rows = [f.result() for f in tech_futs]
        fund_rows = [f.result() for f in fund_futs]
        risk_rows = [f.result() for f in risk_futs]
        comp_rows = [f.result() for f in comp_futs]

        log.info("batch_fetcher.done", stocks=len(symbols))

        tech_table = _to_md_table(tech_rows, [
            "symbol", "price", "rsi", "macd_hist", "score", "trend",
            "support", "resistance", "vol_signal", "mom_1m_pct", "vol_ratio",
        ])
        fund_table = _to_md_table(fund_rows, [
            "symbol", "sector", "pe", "pb", "roe_pct", "roce_pct",
            "d_e", "rev_cagr_3y", "eps_growth_pct", "promoter_pct",
        ])
        risk_table = _to_md_table(risk_rows, [
            "symbol", "entry", "stop", "vol_pct", "beta", "max_dd_pct",
            "circuit_band", "near_circuit", "kelly_pct", "pos_size_pct", "adv_cr",
        ])
        comp_table = _to_md_table(comp_rows, [
            "symbol", "sector", "peers", "peer_count",
            "1m_ret_pct", "3m_ret_pct", "rank_3m", "mcap_rank", "mom_1m_pct",
        ])

        return tech_table, fund_table, risk_table, comp_table
