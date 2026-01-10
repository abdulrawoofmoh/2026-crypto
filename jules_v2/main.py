"""Main script - Test individual strategies"""

import logging
from config.settings import *
from core.data_loader import DataLoader
from core.indicators import add_all_indicators
from core.backtest_engine import BacktestEngine
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from utils.performance import print_results, save_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_trend_strategy(df):
    """Test trend strategy"""
    print("\n" + "="*60)
    print("TESTING TREND STRATEGY")
    print("="*60)

    strategy = TrendStrategy(TREND_PARAMS)

    engine = BacktestEngine(
        TRADING_CONFIG['initial_balance'], TRADING_CONFIG['leverage'],
        TRADING_CONFIG['position_size'], BACKTEST_CONFIG['commission'],
        BACKTEST_CONFIG['slippage']
    )

    results = engine.run(df, strategy)
    print_results(results, 'Trend Strategy')
    save_results(results, 'trend_strategy_results.csv')
    return results

def test_range_strategy(df):
    """Test range strategy"""
    print("\n" + "="*60)
    print("TESTING RANGE STRATEGY")
    print("="*60)

    strategy = RangeStrategy(RANGE_PARAMS)

    engine = BacktestEngine(
        TRADING_CONFIG['initial_balance'], TRADING_CONFIG['leverage'],
        TRADING_CONFIG['position_size'], BACKTEST_CONFIG['commission'],
        BACKTEST_CONFIG['slippage']
    )

    results = engine.run(df, strategy)
    print_results(results, 'Range Strategy')
    save_results(results, 'range_strategy_results.csv')
    return results

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.download_data(
        TRADING_CONFIG['symbol'], TRADING_CONFIG['timeframe'],
        BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date']
    )

    if df.empty:
        print("No data downloaded. Exiting.")
        exit()

    df = add_all_indicators(df, INDICATOR_PARAMS)

    trend_results = test_trend_strategy(df)
    range_results = test_range_strategy(df)

    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print(f"\nTrend Strategy: Win Rate: {trend_results['win_rate']:.1f}% | "
          f"Return: {trend_results['total_return']:.1f}% | Trades: {trend_results['total_trades']}")
    print(f"Range Strategy: Win Rate: {range_results['win_rate']:.1f}% | "
          f"Return: {range_results['total_return']:.1f}% | Trades: {range_results['total_trades']}")
    print("="*60 + "\n")
