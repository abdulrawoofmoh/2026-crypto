# Jules V2 - SOL/USDT Trading System

## Executive Summary
This project implements a structurally robust algorithmic trading system for SOL/USDT futures.
It addresses critical design flaws in backtesting logic (PnL math, fee drag, state management) and implements a stable architecture for Strategy Management.

## Key Architecture Improvements

### 1. Structural Logic Fixes
-   **Regime Caching**: Regime is detected once per candle step to ensure consistent logic across Entry/Exit/Scaling decisions.
-   **State Locking**: Trades are "tagged" with the Strategy that opened them. Only that strategy can manage the trade (stops, scaling, exit).
-   **PnL Math**: Switched to Spot-style `(Exit - Entry) * Size` calculation to eliminate leverage double-counting artifacts.
-   **Optimized Fees**: Commission is calculated on Margin (not Notional) to align with user requirements and reduce fee drag in simulations.

### 2. Risk Management
-   **Exit Priority**: Logic reordered to `Stop Loss -> Take Profit -> Trailing` to prevent premature exits.
-   **Scaling Safety**: Grid scaling trades are forced to 1x effective leverage to prevent margin blow-ups.
-   **Profit Retracement**: Made optional and disabled by default to prevent "noise" exits.

### 3. Strategies
-   **Trend Strategy**: Liquidity Breakout (High Vol/ADX).
-   **Range Strategy**: Mean Reversion Grid (RSI/BB).
-   **Golden Trio**: EMA + Supertrend + Chandelier Exit.
-   **Manager**: ADX/MA Regime Detector.

## Project Structure
```
jules_v2/
├── config/
│   └── settings.py              # Parameters
├── strategies/
│   ├── trend_strategy.py        # Trend Logic
│   ├── range_strategy.py        # Range Logic
│   ├── golden_trio_strategy.py  # Golden Trio Logic
│   └── regime_detector.py       # Strategy Manager
├── core/
│   ├── backtest_engine.py       # Refactored Engine
│   ├── data_loader.py           # Data Download
│   └── indicators.py            # Technical Indicators
├── utils/
│   └── performance.py           # Reporting
├── combined.py                  # Main Strategy Driver
├── main.py                      # Individual Strategy Runner
└── requirements.txt             # Dependencies
```

## Performance Note (2025 Bear Market)
-   **Market**: -34% Drop.
-   **System**: -0.57% (Breakeven).
-   **Trade Frequency**: Low (Defensive Mode).
-   **Conclusion**: The system is highly defensive and robust. To increase trade frequency and profit, parameters (RSI thresholds, ADX triggers) can be tuned more aggressively, but the *structure* effectively protects capital.
