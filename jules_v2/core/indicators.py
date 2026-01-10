"""Technical indicators for trading strategies"""

import pandas as pd
import numpy as np

def add_all_indicators(df, params):
    """Add all technical indicators to dataframe"""
    df = df.copy()

    # Moving Averages
    df['ma_fast'] = df['close'].rolling(window=params['ma_fast']).mean()
    df['ma_slow'] = df['close'].rolling(window=params['ma_slow']).mean()

    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=params['atr_period']).mean()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=params['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params['rsi_period']).mean()
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(100) # If loss is 0, RSI is 100

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=params['bb_period']).mean()
    df['bb_std'] = df['close'].rolling(window=params['bb_period']).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * params['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * params['bb_std'])

    # ADX
    df = calculate_adx(df, params['adx_period'])

    # Volume
    df['volume_ma'] = df['volume'].rolling(window=params['volume_period']).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    return df

def calculate_adx(df, period=14):
    """Calculate ADX indicator"""
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()

    df['plus_dm'] = np.where(
        (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
        df['high_diff'], 0
    )
    df['minus_dm'] = np.where(
        (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
        df['low_diff'], 0
    )

    # Smoothing (Wilder's Smoothing is standard for ADX, but rolling mean is acceptable for approximation)
    # Using Rolling Mean here as per original user code, can upgrade to Wilder if needed.
    df['plus_dm_smooth'] = df['plus_dm'].rolling(window=period).mean()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(window=period).mean()

    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['atr'])

    # Handle division by zero
    sum_di = df['plus_di'] + df['minus_di']
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / sum_di.replace(0, np.nan)
    df['dx'] = df['dx'].fillna(0)

    df['adx'] = df['dx'].rolling(window=period).mean()

    df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm',
             'plus_dm_smooth', 'minus_dm_smooth', 'dx'], axis=1, inplace=True)

    return df
