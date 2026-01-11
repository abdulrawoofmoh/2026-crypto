"""Combined strategy with Strategy Manager and State Awareness"""

import logging
from config.settings import *
from core.data_loader import DataLoader
from core.indicators import add_all_indicators
from core.backtest_engine import BacktestEngine
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.golden_trio_strategy import GoldenTrioStrategy
from strategies.regime_detector import StrategyManager
from utils.performance import print_results, save_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CombinedStrategy:
    """Combined strategy with regime switching and state locking"""

    def __init__(self, trend_strategy, range_strategy, manager):
        self.trend_strategy = trend_strategy
        self.range_strategy = range_strategy
        self.manager = manager
        # Cache for current step
        self.current_regime = None

    def prepare_step(self, df, idx):
        """Calculate and cache regime for this step"""
        self.current_regime = self.manager.detect_regime(df, idx)

    def generate_signal(self, df, idx):
        """Generate signal based on cached regime"""
        # Ensure prepare_step was called, or call it
        if self.current_regime is None:
            self.prepare_step(df, idx)

        regime = self.current_regime
        signal = None
        strategy_name = None

        if regime == 'TRENDING':
            signal = self.trend_strategy.generate_signal(df, idx)
            strategy_name = 'TREND'
        elif regime == 'RANGING':
            signal = self.range_strategy.generate_signal(df, idx)
            strategy_name = 'RANGE'

        if signal:
            # Tag the signal with the strategy name for locking
            signal['strategy'] = strategy_name
            signal['regime'] = regime

        return signal

    def calculate_stops(self, df, idx, signal_data):
        """Calculate stops using the strategy that generated the signal"""
        strategy_name = signal_data.get('strategy')

        if strategy_name == 'TREND':
            return self.trend_strategy.calculate_stops(df, idx, signal_data)
        elif strategy_name == 'RANGE':
            return self.range_strategy.calculate_stops(df, idx, signal_data)

        # Fallback if no strategy tag (shouldn't happen with new engine)
        return self.range_strategy.calculate_stops(df, idx, signal_data)

    def check_scale_in(self, df, idx, position):
        """
        Check for scaling using the strategy that OWNS the position.
        Ignores current regime.
        """
        strategy_name = position.get('strategy')

        if strategy_name == 'TREND':
             if hasattr(self.trend_strategy, 'check_scale_in'):
                 return self.trend_strategy.check_scale_in(df, idx, position)
        elif strategy_name == 'RANGE':
             if hasattr(self.range_strategy, 'check_scale_in'):
                 return self.range_strategy.check_scale_in(df, idx, position)

        return None

def test_combined_strategy():
    """Test combined strategy"""
    print("\n" + "="*60)
    print("TESTING COMBINED STRATEGY (Fixed Architecture)")
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

    combined = CombinedStrategy(trend_strategy, range_strategy, manager)

    engine = BacktestEngine(
        TRADING_CONFIG['initial_balance'],
        TRADING_CONFIG['leverage'],
        TRADING_CONFIG['position_size'],
        BACKTEST_CONFIG['commission'],
        BACKTEST_CONFIG['slippage'],
        profit_retracement_pct=None # DISABLED as per recommendation
    )

    results = engine.run(df, combined)
    print_results(results, 'Combined Strategy')
    save_results(results, 'combined_strategy_results.csv')
    return results

if __name__ == "__main__":
    results = test_combined_strategy()
