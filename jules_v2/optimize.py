"""Optimizer for Jules V2 Trading System - Full Authority Mode"""

import itertools
import pandas as pd
import logging
from config.settings import *
from core.data_loader import DataLoader
from core.indicators import add_all_indicators
from core.backtest_engine import BacktestEngine
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.regime_detector import StrategyManager
from combined import CombinedStrategy

# Disable logging for optimizer to keep output clean
logging.basicConfig(level=logging.WARNING)

def optimize_system(df):
    print("\n" + "="*60)
    print("OPTIMIZING FULL SYSTEM (RISK + STRATEGY)")
    print("="*60)

    # 1. Strategy Parameters (Reduced Grid)
    # Trend
    trend_lookback = [50]
    trend_adx = [20, 25]

    # Range
    range_rsi_os = [25, 30]
    range_rsi_ob = [70, 75]

    # 2. Risk Parameters (The "Full Authority" part)
    leverage_opts = [3, 5, 10]
    pos_size_opts = [0.10, 0.20, 0.30]
    retracement_opts = [0.10, 0.25]

    # Combine
    param_grid = list(itertools.product(
        trend_lookback, trend_adx,
        range_rsi_os, range_rsi_ob,
        leverage_opts, pos_size_opts, retracement_opts
    ))

    print(f"Testing {len(param_grid)} combinations on 3 Months Data...")

    best_return = -9999
    best_sharpe = -9999
    best_config = {}

    # Strategy Manager (Fixed for now as it just switches)
    manager = StrategyManager(REGIME_PARAMS)

    count = 0
    for t_look, t_adx, r_os, r_ob, lev, size, ret in param_grid:
        count += 1
        if count % 10 == 0:
            print(f"Processing {count}/{len(param_grid)}...")

        # Update Strategy Params
        t_params = TREND_PARAMS.copy()
        t_params['lookback_period'] = t_look
        t_params['adx_threshold'] = t_adx

        r_params = RANGE_PARAMS.copy()
        r_params['rsi_oversold'] = r_os
        r_params['rsi_overbought'] = r_ob

        # Instantiate Strategies
        trend_strat = TrendStrategy(t_params)
        range_strat = RangeStrategy(r_params)
        combined = CombinedStrategy(trend_strat, range_strat, manager)

        # Instantiate Engine with Risk Params
        engine = BacktestEngine(
            initial_balance=100,
            leverage=lev,
            base_position_size=size,
            commission=0.0004,
            slippage=0.0005,
            profit_retracement_pct=ret
        )

        results = engine.run(df, combined)

        # Score: We want positive return first, then Sharpe
        score = results['total_return']

        if score > best_return:
            best_return = score
            best_sharpe = results['sharpe_ratio']
            best_config = {
                'trend_adx': t_adx,
                'range_rsi': f"{r_os}/{r_ob}",
                'leverage': lev,
                'pos_size': size,
                'retracement': ret
            }
            print(f"New Best: {best_return:.2f}% (Sharpe: {best_sharpe:.2f}) | {best_config}")

    print(f"\n🏆 BEST CONFIGURATION: {best_config}")
    print(f"Return: {best_return:.2f}%")
    return best_config

if __name__ == "__main__":
    loader = DataLoader()
    # 3 Months Data: Oct 1 2024 to Jan 10 2025
    df = loader.download_data(
        TRADING_CONFIG['symbol'], TRADING_CONFIG['timeframe'],
        '2024-10-01', '2025-01-10'
    )

    if df.empty:
        print("No data. Exiting.")
        exit()

    df = add_all_indicators(df, INDICATOR_PARAMS)

    best_config = optimize_system(df)
