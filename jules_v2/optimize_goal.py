"""Goal Seeking Optimizer for Jules V2 - 2025 Target (Lean)"""

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

logging.basicConfig(level=logging.WARNING)

def run_scenario(df, name, lev, size, steps, step_atr, ret):
    print(f"Testing Scenario: {name} (Lev:{lev}x, Size:{size*100}%, Steps:{steps})")

    manager = StrategyManager(REGIME_PARAMS)
    t_params = TREND_PARAMS.copy()

    r_params = RANGE_PARAMS.copy()
    r_params['scale_in_enabled'] = True
    r_params['max_scale_steps'] = steps
    r_params['scale_step_atr'] = step_atr
    r_params['risk_reward_ratio'] = 0.8 # Tight profit taking for grid

    range_strat = RangeStrategy(r_params)
    trend_strat = TrendStrategy(t_params)
    combined = CombinedStrategy(trend_strat, range_strat, manager)

    engine = BacktestEngine(
        initial_balance=100,
        leverage=lev,
        base_position_size=size,
        commission=0.0004,
        slippage=0.0005,
        profit_retracement_pct=ret
    )

    results = engine.run(df, combined)
    print(f"Result: {results['total_return']:.2f}% (DD: {results['max_drawdown']:.2f}%)")
    return results['total_return'], {
        'leverage': lev, 'pos_size': size, 'scale_steps': steps,
        'scale_step_atr': step_atr, 'retracement': ret
    }

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.download_data(
        TRADING_CONFIG['symbol'], TRADING_CONFIG['timeframe'],
        '2025-01-01', '2025-12-31'
    )
    df = add_all_indicators(df, INDICATOR_PARAMS)

    best_ret = -9999
    best_cfg = {}

    # Scenario A: Conservative Grid (Survival)
    ret, cfg = run_scenario(df, "Conservative", 5, 0.05, 5, 1.5, 0.25)
    if ret > best_ret: best_ret, best_cfg = ret, cfg

    # Scenario B: Aggressive Grid (Growth)
    ret, cfg = run_scenario(df, "Aggressive", 10, 0.05, 3, 1.0, 0.25)
    if ret > best_ret: best_ret, best_cfg = ret, cfg

    # Scenario C: High Leverage Scalp
    ret, cfg = run_scenario(df, "High Lev", 20, 0.02, 5, 1.0, 0.50)
    if ret > best_ret: best_ret, best_cfg = ret, cfg

    print(f"\n🏆 WINNER: {best_ret:.2f}% | {best_cfg}")
