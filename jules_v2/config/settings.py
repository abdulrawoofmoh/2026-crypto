"""Configuration settings for Jules V2 Trading System"""

# API Configuration
API_CONFIG = {
    'exchange': 'binance',
    'api_key': '',
    'api_secret': '',
    'testnet': True,
}

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'SOL/USDT',
    'timeframe': '15m',
    'leverage': 5,           # OPTIMIZED for 2025 Survival
    'position_size': 0.05,   # OPTIMIZED for Grid Scaling (5% start)
    'initial_balance': 100,
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'start_date': '2025-01-01',
    'end_date': '2025-12-31',
    'commission': 0.0004,
    'slippage': 0.0005,
}

# RISK MANAGEMENT
RISK_PARAMS = {
    'profit_retracement_pct': 0.25, # OPTIMIZED: Looser trailing
}

# TREND STRATEGY PARAMETERS
TREND_PARAMS = {
    'lookback_period': 50,
    'volume_multiplier': 1.5,
    'adx_threshold': 20,
    'stop_loss_atr': 2.0,
}

# RANGE STRATEGY PARAMETERS (Optimized for Grid)
RANGE_PARAMS = {
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'bb_period': 20,
    'bb_std': 2,
    'adx_threshold': 20,
    'stop_loss_atr': 5.0,         # Wide stop for grid
    'risk_reward_ratio': 0.8,     # Tight profit taking
    'scale_in_enabled': True,     # ENABLED
    'max_scale_steps': 5,         # OPTIMIZED
    'scale_step_atr': 1.5,        # OPTIMIZED
}

# REGIME DETECTION PARAMETERS
REGIME_PARAMS = {
    'adx_period': 14,
    'trend_threshold': 25,
    'range_threshold': 20,
    'ma_distance_pct': 5,
}

# INDICATOR PARAMETERS
INDICATOR_PARAMS = {
    'ma_fast': 20,
    'ma_slow': 50,
    'atr_period': 14,
    'volume_period': 20,
    'adx_period': 14,
    'bb_period': 20,
    'bb_std': 2,
    'rsi_period': 14,
}
