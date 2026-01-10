"""Technical indicators for trading strategies"""

import pandas as pd
import numpy as np

def add_all_indicators(df, params):
    """Add all technical indicators to dataframe"""
    df = df.copy()

    # Moving Averages
    df['ma_fast'] = df['close'].rolling(window=params.get('ma_fast', 20)).mean()
    df['ma_slow'] = df['close'].rolling(window=params.get('ma_slow', 50)).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=params.get('atr_period', 14)).mean()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=params.get('rsi_period', 14)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params.get('rsi_period', 14)).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(100)

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=params.get('bb_period', 20)).mean()
    df['bb_std'] = df['close'].rolling(window=params.get('bb_period', 20)).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * params.get('bb_std', 2))
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * params.get('bb_std', 2))

    # ADX
    df = calculate_adx(df, params.get('adx_period', 14))

    # Volume
    df['volume_ma'] = df['volume'].rolling(window=params.get('volume_period', 20)).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Supertrend
    st_period = params.get('supertrend_period', 10)
    st_multiplier = params.get('supertrend_multiplier', 3)
    df = calculate_supertrend(df, st_period, st_multiplier)

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

    df['plus_dm_smooth'] = df['plus_dm'].rolling(window=period).mean()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(window=period).mean()

    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['atr'])

    sum_di = df['plus_di'] + df['minus_di']
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / sum_di.replace(0, np.nan)
    df['dx'] = df['dx'].fillna(0)

    df['adx'] = df['dx'].rolling(window=period).mean()

    df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm',
             'plus_dm_smooth', 'minus_dm_smooth', 'dx'], axis=1, inplace=True)

    return df

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate Supertrend Indicator"""
    # Calculate ATR for Supertrend if different period needed, but usually we use the common ATR
    # For correctness with specific period, let's recalc ATR local or rely on df['atr'] if periods match.
    # We'll compute basic HL2
    hl2 = (df['high'] + df['low']) / 2

    # Re-calculate ATR for supertrend specific period
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(window=period).mean()

    # Basic Bands
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Initialize Final Bands
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index) # 1: Bullish, -1: Bearish

    # Iterative calculation (Supertrend is path-dependent)
    # Using numpy arrays for speed
    close = df['close'].values
    bu = basic_upper.values
    bl = basic_lower.values
    fu = np.zeros(len(df))
    fl = np.zeros(len(df))
    st = np.zeros(len(df))
    tr_dir = np.zeros(len(df)) # Trend direction

    # Start loop
    for i in range(1, len(df)):
        # Upper Band
        if bu[i] < fu[i-1] or close[i-1] > fu[i-1]:
            fu[i] = bu[i]
        else:
            fu[i] = fu[i-1]

        # Lower Band
        if bl[i] > fl[i-1] or close[i-1] < fl[i-1]:
            fl[i] = bl[i]
        else:
            fl[i] = fl[i-1]

        # Trend
        if st[i-1] == fu[i-1]: # Previous trend was Bearish (Upper Band active)
            if close[i] > fu[i]:
                tr_dir[i] = 1 # Change to Bullish
                st[i] = fl[i]
            else:
                tr_dir[i] = -1 # Stay Bearish
                st[i] = fu[i]
        else: # Previous trend was Bullish (Lower Band active)
            if close[i] < fl[i]:
                tr_dir[i] = -1 # Change to Bearish
                st[i] = fu[i]
            else:
                tr_dir[i] = 1 # Stay Bullish
                st[i] = fl[i]

    df['supertrend'] = st
    df['supertrend_direction'] = tr_dir # 1 = Bullish, -1 = Bearish

    return df
