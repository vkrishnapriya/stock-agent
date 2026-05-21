# Entry Point Calculation — Design & Implementation Plan

## Background

The original `_build_buy_candidate()` logic in `src/workflows/buyer_workflow.py` set the
entry zone to the **support level** (a historical MA/pivot below current price).

For momentum stocks that have already run up, this means:
- The entry zone is in the past (current price is already above it)
- The current price sits between the entry and target → little upside remaining
- Reward-to-Risk (Reward:Risk) can be below 1:1, making the trade unviable

This document captures the requirements, design decisions, and implementation plan to fix this.

---

## Requirements (from `entry_point_requirements.md`)

### Technical Analysis Methods
- **Support Levels** — price tends to bounce from historical support zones; natural buy zones for pullback entries
- **Moving Averages** — buy when price pulls back to SMA50 or SMA200; a golden cross (SMA50 crossing above SMA200) signals entry
- **RSI** — RSI below 30 = oversold → potential buy signal; best combined with support levels
- **Breakout Entry** — enter when stock breaks above a key resistance level with strong volume; common in momentum trading

### Fundamental Analysis Methods
- **P/E Ratio** — buy when P/E is below historical mean (undervaluation signal)
- **DCF / Intrinsic Value** — enter when market price is significantly below intrinsic value (margin of safety)
- **52-Week Low Strategy** — stocks near 52-week low may be undervalued; always verify the reason first

### Risk-Based Entry (Most Practical)
1. Identify the **stop-loss** first — the price at which you exit if wrong (e.g., below a key support level)
2. Calculate **risk per share** = Entry Price − Stop Loss Price
3. Set **position size** based on capital at risk (e.g., 1–2% of portfolio)
4. Enter only where **Reward-to-Risk ratio is at least 2:1 or 3:1**

> Example: Entry at Rs 100, Stop Loss at Rs 95, Target at Rs 115
> Risk per share = Rs 5, Reward per share = Rs 15 → Reward-to-Risk = 3:1 (acceptable)

### Quick Reference by Trading Goal

| Goal | Best Method |
|---|---|
| Long-term investing | DCF + P/E + DCA |
| Swing trading | Support levels + Moving averages |
| Momentum trading | Breakout + Volume confirmation |
| Risk management | Stop-loss based entry |

The most reliable entries combine **at least one technical signal** with **a fundamental reason** to own the stock. Never enter based on a single indicator alone.

---

## Problem Statement

Current code (`buyer_workflow.py:_build_buy_candidate`):

```python
support = technical.support_level_inr        # historical MA/pivot below current price
entry_lower = round(support, 2)              # entry = support  ← WRONG for momentum stocks
entry_upper = round(support * 1.02, 2)
stop = round(min(stop, entry_lower * 0.97))  # 3% below support
target = round(max(resistance, entry_upper * 1.05), 2)  # only 5% above entry
```

For a stock at Rs 1,700 with support Rs 1,522 and resistance Rs 1,775:
- Entry zone: Rs 1,522–1,553 (stock is already above this — not executable)
- Stop: Rs 1,477
- Target: Rs 1,775 (only ~4% above current price)
- Reward-to-Risk: (1,775 − 1,700) / (1,700 − 1,477) = 75 / 223 ≈ **0.34:1** (unacceptable)

---

## Design: Three Entry Modes

Entry mode is selected automatically based on RSI, volume signal, and price position relative to support/resistance. All three modes anchor to the **current price** as the baseline.

---

### Mode A — BREAKOUT

**When to use:** Momentum trading — stock is breaking out of a resistance zone with volume confirmation.

**Conditions:**
- `volume_signal == "HIGH"` (OBV rising, above-average volume)
- `last_price >= resistance * 0.98` (price at or just below resistance, i.e. attempting breakout)

**Entry zone:** `[last_price, last_price × 1.005]`
Buy the breakout as it happens; the 0.5% band accommodates limit orders.

**Rationale from requirements:** *"Enter when stock breaks above key resistance level with strong volume — common in momentum trading."*

---

### Mode B — PULLBACK

**When to use:** Swing trading — stock has pulled back to a key support level and RSI signals oversold conditions.

**Conditions:**
- `rsi < 45` (price momentum weakening, not overbought)
- `last_price <= support * 1.03` (price within 3% of support level)

**Entry zone:** `[support, support × 1.02]`
This is the only mode where the support level is used directly as the entry.

**Rationale from requirements:** *"Buy when price pulls back to the 50-day or 200-day moving average."*
*"RSI below 30 = oversold → potential buy signal; best used in combination with support levels."*

---

### Mode C — CURRENT PRICE (default)

**When to use:** Default for all other cases — trending stock with momentum but no active breakout or pullback signal.

**Conditions:** Everything else (RSI 45–70, price between support and resistance, NORMAL volume).

**Entry zone:** `[last_price, last_price × 1.005]`

**Rationale from requirements:** Risk-based entry at current market price with stop set at nearest support.

---

## Stop-Loss Selection

Applied in priority order after entry zone is determined:

1. Use `risk.suggested_stop_loss_inr` (from Risk Management agent) if:
   - It is below `entry_lower`
   - It is not more than 8% below `entry_lower`
2. Else use `support_level_inr` if it is below `entry_lower`
3. Else fallback: `entry_lower × 0.95` (5% hard floor)

**Hard cap:** Stop is never placed more than 8% below entry — enforces position sizing discipline and limits drawdown on any single trade.

---

## Target Price with Reward-to-Risk Enforcement

This is the most critical fix. Target is no longer just the resistance level — it must satisfy a minimum Reward-to-Risk ratio.

```
risk_per_share      = entry_lower − stop
min_target (2:1 RR) = entry_upper + 2 × risk_per_share
target              = max(resistance_level_inr, min_target)
```

**Reward-to-Risk ratio (Reward:Risk)** = (Target − Entry) / (Entry − Stop)

- A ratio of **2:1** means for every Rs 1 risked, you expect Rs 2 of reward
- A ratio of **3:1** is ideal for swing/momentum trades
- The system enforces a **minimum of 2:1** — candidates below this threshold are discarded

### Discard Rule

If after target calculation:
```
(target − entry_upper) / (entry_lower − stop) < 1.5
```
Return `None` — the stock does not offer enough upside relative to the risk taken.
This happens when resistance is too close to the current price (e.g., stock near 52-week high with no room to run).

---

## Before vs After Example

Stock: TATACOMM, Current price Rs 1,700, Support Rs 1,522, Resistance Rs 1,775, RSI 62, Volume NORMAL → **Mode C (Current Price)**

| Metric | Before (broken) | After (Mode C) |
|---|---|---|
| Entry zone | Rs 1,522 – Rs 1,553 (already passed) | Rs 1,700 – Rs 1,708 (executable now) |
| Stop loss | Rs 1,477 (3% below support) | Rs 1,522 (nearest support below entry) |
| Risk per share | Rs 45 (entry to stop) | Rs 178 (entry to stop) |
| Target | Rs 1,775 (only 4.4% above current) | Rs max(1,775, 1,708 + 2×178) = Rs 2,064 |
| Reward-to-Risk | (1,775−1,700)/(1,700−1,477) = **0.34:1** | (2,064−1,708)/(1,708−1,522) = **1.91:1** ≈ 2:1 |
| Executable? | No — entry in the past | Yes — buy at market |

---

## Files to Change

| File | Change | Scope |
|---|---|---|
| `src/models/recommendations.py` | Add `entry_type: Literal["BREAKOUT", "PULLBACK", "CURRENT_PRICE"]` field to `BuyCandidate` | ~3 lines |
| `src/workflows/buyer_workflow.py` | Replace entry/stop/target block in `_build_buy_candidate()` with three-mode logic + Reward-to-Risk enforcement | ~30 lines |

No changes needed to agents, tools, or data models — `TechnicalSignal` already carries `rsi`, `volume_signal`, `support_level_inr`, and `resistance_level_inr`.

---

## Reward-to-Risk Terminology

Throughout this codebase, **Reward-to-Risk** is expressed as **Reward:Risk** (e.g., 2:1).

- **2:1 Reward:Risk** means: for every Rs 1 at risk (entry minus stop), the expected reward is Rs 2 (target minus entry)
- This is the inverse of the commonly misused "Risk:Reward" phrasing
- A 2:1 Reward:Risk trade is acceptable; 3:1 is ideal; anything below 1.5:1 is rejected

> Example: Entry Rs 100, Stop Rs 95, Target Rs 110
> Risk = Rs 5, Reward = Rs 10 → Reward:Risk = 10:5 = **2:1** (acceptable minimum)
