"""
Jules Trading System - Configuration Settings
"""

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'SOL/USDT',
    'timeframe_primary': '15m',      # Main trading timeframe
    'timeframe_trend': '1h',         # Trend filter
    'timeframe_major': '4h',         # Major trend
    'leverage': 20,
    'position_size': 0.5,            # 50% of available balance
    'initial_balance': 50,           # Starting balance in USDT
}

# API Configuration (Leave empty for backtesting)
API_CONFIG = {
    'exchange': 'binance',
    'api_key': '',                   # Mrfq1Em4l1VLliRS8wI7WfdLyzi9w3DewgRFMLa2qLVLAmzbxP7vNkJH6C0bS0b7
    'api_secret': '',                # 75QMULXfXrbd7IlmQYNh910DYtdA55NOsoTPEINU2EQu9zP6UpD3Zn2JAhX4lqTE
    'testnet': True,                 # True for testnet, False for live trading
}

# Strategy Parameters
STRATEGY_PARAMS = {
    # Technical Indicators
    'rsi_period': 14,
    'ma_fast': 20,
    'ma_slow': 50,
    'atr_period': 14,
    'volume_period': 20,
    
    # Signal Thresholds
    'volume_threshold': 1.2,         # Volume must be 1.2x average
    'trend_strength_min': 0.5,       # Minimum trend strength percentage
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'rsi_long_zone': (30, 50),       # RSI range for LONG entries
    'rsi_short_zone': (50, 70),      # RSI range for SHORT entries
    
    # Risk Management
    'sl_atr_mult': 1.5,              # Stop loss at 1.5x ATR
    'tp_ratio': 2.5,                 # Take profit ratio (2.5:1)
    'trailing_stop_activation': 1.5, # Activate trailing at 1.5% profit
    'trailing_stop_distance': 0.8,   # Trail 0.8% behind peak
    'max_trades_per_day': 5,
    'max_consecutive_losses': 3,
    
    # Liquidity Detection
    'order_book_levels': 20,         # Number of order book levels to analyze
    'order_book_imbalance_threshold': 1.5,  # Bid/Ask ratio threshold
    'large_order_multiplier': 5,     # Orders 5x larger than average
    'swing_cluster_distance': 0.02,  # 2% distance for swing clusters
    'volume_profile_bins': 50,       # Number of price bins for volume profile
    
    # Smart Money Concepts
    'bos_lookback': 20,              # Candles to look back for BOS
    'liquidity_sweep_wick_ratio': 0.6,  # Wick must be 60% of candle
    'order_block_strength': 1.5,     # ATR multiplier for order block detection
}

# Probability Scoring Weights
SCORING_WEIGHTS = {
    'trend_15m': 8,
    'trend_1h': 10,
    'trend_4h': 7,
    'order_book_imbalance': 8,
    'volume_node': 7,
    'order_block': 5,
    'fair_value_gap': 5,
    'break_of_structure': 10,
    'liquidity_sweep': 12,
    'displacement': 3,
    'rsi_zone': 5,
    'volume_surge': 5,
    'rsi_divergence': 5,
    'key_level': 5,
    'tight_stop': 5,
}

# Scoring Thresholds
PROBABILITY_THRESHOLDS = {
    'minimum_score': 35,             # Minimum score to take trade
    'strong_signal': 50,             # Strong signal threshold
    'elite_signal': 65,              # Elite signal threshold
}

# Penalties
SCORING_PENALTIES = {
    'against_4h_trend': -15,
    'low_volume': -10,
    'choppy_price': -8,
    'recent_losses': -5,
}

# Data Settings
DATA_CONFIG = {
    'historical_years': 3,
    'candles_lookback': 500,
    'cache_enabled': True,
    'cache_directory': 'data/cache/',
}

# Logging Settings
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_to_file': True,
    'log_directory': 'logs/',
    'log_trades': True,
    'log_signals': True,
}

# Backtest Settings
BACKTEST_CONFIG = {
    'commission': 0.0004,            # 0.04% per trade (Binance futures)
    'slippage': 0.0005,              # 0.05% slippage
    'starting_balance': 50,
    'max_positions': 1,              # Only 1 position at a time
}