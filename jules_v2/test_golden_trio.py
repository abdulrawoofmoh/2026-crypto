"""Test Golden Trio Strategy on 2025 Data"""

import pandas as pd
import logging
import itertools
from config.settings import *
from core.data_loader import DataLoader
from core.indicators import add_all_indicators
from core.backtest_engine import BacktestEngine
from strategies.golden_trio_strategy import GoldenTrioStrategy
from utils.performance import print_results, save_results

logging.basicConfig(level=logging.WARNING)

def run_test():
    print("\n" + "="*60)
    print("TESTING GOLDEN TRIO STRATEGY (2025)")
    print("="*60)

    loader = DataLoader()
    df = loader.download_data(
        TRADING_CONFIG['symbol'], TRADING_CONFIG['timeframe'],
        BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date']
    )

    if df.empty:
        print("No data.")
        return

    # Parameter Grid
    st_periods = [10, 14]
    st_mults = [3]
    adx_thresholds = [20, 25]
    chandelier_mults = [2.5, 3.0, 3.5]

    param_grid = list(itertools.product(st_periods, st_mults, adx_thresholds, chandelier_mults))
    print(f"Testing {len(param_grid)} parameter combinations...")

    best_ret = -9999
    best_cfg = {}
    best_results = None

    for period, mult, adx, chand in param_grid:
        # Update Indicator Params
        ind_params = INDICATOR_PARAMS.copy()
        ind_params['supertrend_period'] = period
        ind_params['supertrend_multiplier'] = mult

        # Calculate Indicators (Need to recalc if params change)
        # Optimization Tip: In a real heavy loop, we'd pre-calc all variants columns.
        # Here we just re-run for simplicity as dataset isn't huge.
        df_test = add_all_indicators(df, ind_params)

        # Update Strategy Params
        strat_params = GOLDEN_TRIO_PARAMS.copy()
        strat_params['adx_threshold'] = adx
        strat_params['chandelier_mult'] = chand

        strategy = GoldenTrioStrategy(strat_params)

        engine = BacktestEngine(
            TRADING_CONFIG['initial_balance'],
            TRADING_CONFIG['leverage'],
            TRADING_CONFIG['position_size'],
            BACKTEST_CONFIG['commission'],
            BACKTEST_CONFIG['slippage']
        )

        results = engine.run(df_test, strategy)

        print(f"ST({period},{mult}) ADX:{adx} Chand:{chand} -> Ret: {results['total_return']:.2f}% | Win: {results['win_rate']:.1f}%")

        if results['total_return'] > best_ret:
            best_ret = results['total_return']
            best_cfg = {'st_period': period, 'st_mult': mult, 'adx': adx, 'chandelier': chand}
            best_results = results

    print("\n" + "="*60)
    print(f"🏆 BEST GOLDEN TRIO CONFIG: {best_cfg}")
    print("="*60)
    print_results(best_results, "Best Golden Trio")
    save_results(best_results, "golden_trio_results.csv")

if __name__ == "__main__":
    run_test()
