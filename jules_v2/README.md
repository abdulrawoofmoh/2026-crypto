# Jules V2 - SOL/USDT Trading System

## Executive Summary
This project implements an advanced algorithmic trading system for SOL/USDT futures. It features multiple strategy modules, including a "Golden Trio" Trend Follower and a robust Mean Reversion Grid system.

## Key Features

### 1. Dynamic Strategy Manager
-   Analyzes market conditions (ADX, MA Deviation, Volatility) to select the optimal strategy.
-   Prioritizes **Range Strategy** in choppy markets and **Trend Strategy** in strong momentum regimes.

### 2. "Smart" Risk Management
-   **Base Sizing**: 5% of Account Balance (Optimized for Survival).
-   **Leverage**: 5x (Optimized for Survival).
-   **Dynamic Allocation**: Position size is adjusted based on Confidence Score.
-   **Compounding**: Order value automatically compounds.

### 3. Strategies

#### A. Golden Trio (Trend Follower)
-   **Logic**: Classic Trend Following.
-   **Filter**: Price > 200 EMA (Long bias).
-   **Trigger**: Supertrend (14, 3) Flip.
-   **Confirmation**: ADX > 25.
-   **Exit**: Chandelier Exit (3.5 ATR Trailing Stop).
-   **Performance**: struggles in choppy bear markets (-43% in 2025 backtest).

#### B. Range Strategy (Mean Reversion + Grid)
-   **Entry**: RSI < 30 / > 70 + Bollinger Band Bounce.
-   **Grid Logic**: Adds to position if price moves against (1.5 ATR step) up to 5 times.
-   **Performance**: Excellent survival. (-3.6% in 2025 backtest vs -34% Market).

#### C. Trend Strategy (Liquidity Breakout)
-   **Entry**: Breakout of 50-period Swing Highs/Lows with Volume.

### 4. Optimization
-   Includes `optimize.py`, `optimize_goal.py`, and `test_golden_trio.py` for parameter tuning.

## Project Structure
```
jules_v2/
├── config/
│   └── settings.py              # Parameters
├── strategies/
│   ├── golden_trio_strategy.py  # New Trend Strategy
│   ├── trend_strategy.py        # Liquidity Breakout Logic
│   ├── range_strategy.py        # Mean Reversion Logic
│   └── regime_detector.py       # Strategy Manager
├── core/
│   ├── backtest_engine.py       # Engine with Chandelier/Percent Trailing
│   ├── data_loader.py           # Data Download
│   └── indicators.py            # Supertrend, ADX, ATR, etc.
├── utils/
│   └── performance.py           # Reporting
├── main.py                      # Run individual strategies
├── combined.py                  # Run combined system
├── test_golden_trio.py          # Run Golden Trio Backtest
└── requirements.txt             # Dependencies
```

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Golden Trio Backtest**:
    ```bash
    python test_golden_trio.py
    ```

3.  **Run Combined System**:
    ```bash
    python combined.py
    ```
