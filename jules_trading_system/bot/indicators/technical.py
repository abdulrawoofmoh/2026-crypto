"""
Technical Indicators
RSI, Moving Averages, ATR, Volume analysis, etc.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period
        
    Returns:
        Series with RSI values
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_moving_averages(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate fast and slow moving averages
    
    Args:
        df: DataFrame with 'close' column
        fast_period: Fast MA period
        slow_period: Slow MA period
        
    Returns:
        Tuple of (fast_ma, slow_ma)
    """
    fast_ma = df['close'].rolling(window=fast_period).mean()
    slow_ma = df['close'].rolling(window=slow_period).mean()
    
    return fast_ma, slow_ma


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period
        
    Returns:
        Series with ATR values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate volume ratio (current volume / average volume)
    
    Args:
        df: DataFrame with 'volume' column
        period: Period for average volume
        
    Returns:
        Series with volume ratio
    """
    volume_ma = df['volume'].rolling(window=period).mean()
    volume_ratio = df['volume'] / volume_ma
    
    return volume_ratio


def calculate_trend_strength(df: pd.DataFrame, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """
    Calculate trend strength as percentage difference between MAs
    
    Args:
        df: DataFrame with 'close' column
        fast_ma: Fast moving average
        slow_ma: Slow moving average
        
    Returns:
        Series with trend strength percentage
    """
    trend_strength = np.abs(fast_ma - slow_ma) / df['close'] * 100
    
    return trend_strength


def calculate_momentum(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """
    Calculate price momentum
    
    Args:
        df: DataFrame with 'close' column
        period: Lookback period
        
    Returns:
        Series with momentum percentage
    """
    momentum = df['close'].pct_change(period) * 100
    
    return momentum


def detect_rsi_divergence(df: pd.DataFrame, rsi: pd.Series, lookback: int = 14) -> pd.Series:
    """
    Detect RSI divergence (bullish or bearish)
    
    Args:
        df: DataFrame with 'close' column
        rsi: RSI series
        lookback: Lookback period
        
    Returns:
        Series with divergence signals (1=bullish, -1=bearish, 0=none)
    """
    divergence = pd.Series(0, index=df.index)
    
    for i in range(lookback, len(df)):
        # Get recent price and RSI
        recent_price = df['close'].iloc[i-lookback:i+1]
        recent_rsi = rsi.iloc[i-lookback:i+1]
        
        # Find local extremes
        price_min_idx = recent_price.idxmin()
        price_max_idx = recent_price.idxmax()
        rsi_min_idx = recent_rsi.idxmin()
        rsi_max_idx = recent_rsi.idxmax()
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (price_min_idx == i and rsi_min_idx != i and 
            recent_rsi.iloc[-1] > recent_rsi.min()):
            divergence.iloc[i] = 1
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (price_max_idx == i and rsi_max_idx != i and 
            recent_rsi.iloc[-1] < recent_rsi.max()):
            divergence.iloc[i] = -1
    
    return divergence


def calculate_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate all technical indicators at once
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary with indicator parameters
        
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = calculate_rsi(df, params.get('rsi_period', 14))
    
    # Moving Averages
    df['ma_fast'], df['ma_slow'] = calculate_moving_averages(
        df, 
        params.get('ma_fast', 20), 
        params.get('ma_slow', 50)
    )
    
    # ATR
    df['atr'] = calculate_atr(df, params.get('atr_period', 14))
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=params.get('volume_period', 20)).mean()
    df['volume_ratio'] = calculate_volume_ratio(df, params.get('volume_period', 20))
    
    # Trend Strength
    df['trend_strength'] = calculate_trend_strength(df, df['ma_fast'], df['ma_slow'])
    
    # Momentum
    df['momentum'] = calculate_momentum(df, period=5)
    
    # RSI Divergence
    df['rsi_divergence'] = detect_rsi_divergence(df, df['rsi'], lookback=14)
    
    return df


def identify_trend(df: pd.DataFrame, idx: int) -> str:
    """
    Identify current trend direction
    
    Args:
        df: DataFrame with indicators
        idx: Current index
        
    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    if idx < 1:
        return 'neutral'
    
    row = df.iloc[idx]
    
    # Bullish: price > fast MA > slow MA
    if row['close'] > row['ma_fast'] > row['ma_slow']:
        return 'bullish'
    
    # Bearish: price < fast MA < slow MA
    elif row['close'] < row['ma_fast'] < row['ma_slow']:
        return 'bearish'
    
    else:
        return 'neutral'


def check_trend_alignment(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, idx: int) -> dict:
    """
    Check if trends are aligned across multiple timeframes
    
    Args:
        df_15m: 15-minute DataFrame
        df_1h: 1-hour DataFrame  
        df_4h: 4-hour DataFrame
        idx: Current index in 15m timeframe
        
    Returns:
        Dictionary with trend alignment info
    """
    trend_15m = identify_trend(df_15m, idx)
    
    # Find corresponding candle in higher timeframes
    current_time = df_15m.iloc[idx]['timestamp']
    
    # Get closest 1h candle
    idx_1h = (df_1h['timestamp'] - current_time).abs().idxmin()
    trend_1h = identify_trend(df_1h, idx_1h)
    
    # Get closest 4h candle
    idx_4h = (df_4h['timestamp'] - current_time).abs().idxmin()
    trend_4h = identify_trend(df_4h, idx_4h)
    
    # Check alignment
    all_bullish = (trend_15m == 'bullish' and trend_1h == 'bullish' and trend_4h == 'bullish')
    all_bearish = (trend_15m == 'bearish' and trend_1h == 'bearish' and trend_4h == 'bearish')
    
    return {
        'trend_15m': trend_15m,
        'trend_1h': trend_1h,
        'trend_4h': trend_4h,
        'aligned': all_bullish or all_bearish,
        'direction': 'bullish' if all_bullish else ('bearish' if all_bearish else 'mixed')
    }