"""Combined strategy with Strategy Manager"""

import logging
from config.settings import *
from core.data_loader import DataLoader
from core.indicators import add_all_indicators
from core.backtest_engine import BacktestEngine
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.regime_detector import StrategyManager
from utils.performance import print_results, save_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CombinedStrategy:
    """Combined strategy with regime switching"""

    def __init__(self, trend_strategy, range_strategy, manager):
        self.trend_strategy = trend_strategy
        self.range_strategy = range_strategy
        self.manager = manager

    def generate_signal(self, df, idx):
        """Generate signal based on regime"""
        regime = self.manager.detect_regime(df, idx)

        if regime == 'TRENDING':
            return self.trend_strategy.generate_signal(df, idx)
        elif regime == 'RANGING':
            return self.range_strategy.generate_signal(df, idx)
        return None

    def calculate_stops(self, df, idx, signal_data):
        """Calculate stops based on regime"""
        regime = self.manager.detect_regime(df, idx)

        if regime == 'TRENDING':
            return self.trend_strategy.calculate_stops(df, idx, signal_data)
        else:
            return self.range_strategy.calculate_stops(df, idx, signal_data)

    def check_scale_in(self, df, idx, position):
        """
        Check for scaling based on active strategy logic (Regime-dependent).
        This delegates the 'Grid' capability to the underlying strategies.
        """
        regime = self.manager.detect_regime(df, idx)

        if regime == 'TRENDING':
             # Trend strategy typically doesn't scale, but check just in case
             if hasattr(self.trend_strategy, 'check_scale_in'):
                 return self.trend_strategy.check_scale_in(df, idx, position)
        elif regime == 'RANGING':
             # Range strategy definitely uses scaling (Grid Logic)
             if hasattr(self.range_strategy, 'check_scale_in'):
                 return self.range_strategy.check_scale_in(df, idx, position)
        return None

def test_combined_strategy():
    """Test combined strategy"""
    print("\n" + "="*60)
    print("TESTING COMBINED STRATEGY")
    print("="*60)

    loader = DataLoader()
    df = loader.download_data(
        TRADING_CONFIG['symbol'], TRADING_CONFIG['timeframe'],
        BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date']
    )

    if df.empty:
        print("No data. Exiting.")
        return

    df = add_all_indicators(df, INDICATOR_PARAMS)

    trend_strategy = TrendStrategy(TREND_PARAMS)
    range_strategy = RangeStrategy(RANGE_PARAMS)
    manager = StrategyManager(REGIME_PARAMS)

    # Just for statistics, print regime distribution
    df_regime = manager.add_regime_column(df.copy())
    regime_counts = df_regime['regime'].value_counts()
    print(f"\nMarket Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

    combined = CombinedStrategy(trend_strategy, range_strategy, manager)

    engine = BacktestEngine(
        TRADING_CONFIG['initial_balance'], TRADING_CONFIG['leverage'],
        TRADING_CONFIG['position_size'], BACKTEST_CONFIG['commission'],
        BACKTEST_CONFIG['slippage']
    )

    results = engine.run(df, combined)
    print_results(results, 'Combined Strategy')
    save_results(results, 'combined_strategy_results.csv')
    return results

if __name__ == "__main__":
    results = test_combined_strategy()
