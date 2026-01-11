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
    'leverage': 5,           # Safety First
    'position_size': 0.04,   # Initial Entry 4%
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
    'profit_retracement_pct': None, # DISABLED (as per Phase 2 Plan)
}

# GOLDEN TRIO STRATEGY PARAMETERS
GOLDEN_TRIO_PARAMS = {
    'adx_threshold': 25,
    'chandelier_mult': 3.0,
    'supertrend_period': 10,
    'supertrend_multiplier': 3,
}

# TREND STRATEGY PARAMETERS
TREND_PARAMS = {
    'lookback_period': 50,
    'volume_multiplier': 1.5,
    'adx_threshold': 20,
    'stop_loss_atr': 2.0,
}

# RANGE STRATEGY PARAMETERS (Phase 2)
RANGE_PARAMS = {
    'rsi_period': 14,
    'rsi_oversold': 40,           # Frequency: 40
    'rsi_overbought': 60,         # Frequency: 60
    'bb_period': 20,
    'bb_std': 2,
    'adx_threshold': 30,          # Only range if trend is weak? No, user said ADX < 20 for ranging logic in general but here we force RSI. Let's stick to default/loose.
    'stop_loss_atr': 1.2,         # Adaptive Stop
    'risk_reward_ratio': 0.0,     # Dynamic Exit
    'scale_in_enabled': True,
    'max_scale_steps': 2,
    'scale_step_atr': 0.0,        # Controlled by RSI
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
    'supertrend_period': 10,
    'supertrend_multiplier': 3,
}
