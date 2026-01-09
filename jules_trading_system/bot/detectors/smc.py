"""
Smart Money Concepts (SMC) Detector
Break of Structure, Change of Character, Liquidity Sweeps, Order Blocks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_break_of_structure(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS)
    
    Args:
        df: DataFrame with OHLCV and swing points
        lookback: Candles to look back for swing points
        
    Returns:
        DataFrame with BOS markers
    """
    df = df.copy()
    df['bos_bullish'] = False
    df['bos_bearish'] = False
    df['bos_price'] = np.nan
    
    # First detect swing points if not already done
    if 'swing_high' not in df.columns:
        from .liquidity import detect_swing_points
        df = detect_swing_points(df, lookback=5)
    
    for i in range(lookback, len(df)):
        current_price = df.iloc[i]['close']
        
        # Look back for last significant swing high
        recent_df = df.iloc[max(0, i-lookback):i]
        swing_highs = recent_df[recent_df['swing_high'] == True]
        
        if not swing_highs.empty:
            last_swing_high = swing_highs.iloc[-1]['high']
            
            # Bullish BOS: price breaks above last swing high
            if current_price > last_swing_high:
                df.loc[df.index[i], 'bos_bullish'] = True
                df.loc[df.index[i], 'bos_price'] = last_swing_high
        
        # Look back for last significant swing low
        swing_lows = recent_df[recent_df['swing_low'] == True]
        
        if not swing_lows.empty:
            last_swing_low = swing_lows.iloc[-1]['low']
            
            # Bearish BOS: price breaks below last swing low
            if current_price < last_swing_low:
                df.loc[df.index[i], 'bos_bearish'] = True
                df.loc[df.index[i], 'bos_price'] = last_swing_low
    
    return df


def detect_change_of_character(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect Change of Character (CHOCH) - potential trend reversal
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Candles to look back
        
    Returns:
        DataFrame with CHOCH markers
    """
    df = df.copy()
    df['choch_bullish'] = False
    df['choch_bearish'] = False
    
    # Detect BOS first if not done
    if 'bos_bullish' not in df.columns:
        df = detect_break_of_structure(df, lookback)
    
    for i in range(lookback, len(df)):
        recent_df = df.iloc[max(0, i-lookback):i+1]
        
        # Count recent bullish and bearish BOS
        bullish_bos_count = recent_df['bos_bullish'].sum()
        bearish_bos_count = recent_df['bos_bearish'].sum()
        
        # CHOCH occurs when trend changes
        # Bullish CHOCH: was bearish, now bullish BOS
        if df.iloc[i]['bos_bullish'] and bearish_bos_count > bullish_bos_count:
            df.loc[df.index[i], 'choch_bullish'] = True
        
        # Bearish CHOCH: was bullish, now bearish BOS
        if df.iloc[i]['bos_bearish'] and bullish_bos_count > bearish_bos_count:
            df.loc[df.index[i], 'choch_bearish'] = True
    
    return df


def detect_liquidity_sweep(df: pd.DataFrame, wick_ratio: float = 0.6, lookback: int = 10) -> pd.DataFrame:
    """
    Detect liquidity sweeps (stop hunts with rejection)
    
    Args:
        df: DataFrame with OHLCV data
        wick_ratio: Minimum wick size as ratio of total candle
        lookback: Candles to look back for swing points
        
    Returns:
        DataFrame with liquidity sweep markers
    """
    df = df.copy()
    df['sweep_bullish'] = False
    df['sweep_bearish'] = False
    df['sweep_price'] = np.nan
    
    # Detect swing points if not done
    if 'swing_high' not in df.columns:
        from .liquidity import detect_swing_points
        df = detect_swing_points(df, lookback=5)
    
    for i in range(lookback, len(df)):
        candle = df.iloc[i]
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            continue
        
        # Calculate wick sizes
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
        
        # Look for recent swing highs/lows
        recent_df = df.iloc[max(0, i-lookback):i]
        
        # Bullish sweep: takes out swing low then reverses up (long lower wick)
        if lower_wick_ratio >= wick_ratio:
            swing_lows = recent_df[recent_df['swing_low'] == True]
            if not swing_lows.empty:
                last_swing_low = swing_lows.iloc[-1]['low']
                
                # Check if this candle went below swing low but closed above
                if candle['low'] <= last_swing_low and candle['close'] > last_swing_low:
                    df.loc[df.index[i], 'sweep_bullish'] = True
                    df.loc[df.index[i], 'sweep_price'] = last_swing_low
        
        # Bearish sweep: takes out swing high then reverses down (long upper wick)
        if upper_wick_ratio >= wick_ratio:
            swing_highs = recent_df[recent_df['swing_high'] == True]
            if not swing_highs.empty:
                last_swing_high = swing_highs.iloc[-1]['high']
                
                # Check if this candle went above swing high but closed below
                if candle['high'] >= last_swing_high and candle['close'] < last_swing_high:
                    df.loc[df.index[i], 'sweep_bearish'] = True
                    df.loc[df.index[i], 'sweep_price'] = last_swing_high
    
    return df


def detect_order_blocks(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Detect order blocks (institutional entry zones)
    
    Args:
        df: DataFrame with OHLCV data
        atr_mult: ATR multiplier for strong moves
        
    Returns:
        DataFrame with order block markers
    """
    df = df.copy()
    df['ob_bullish'] = False
    df['ob_bearish'] = False
    df['ob_price_top'] = np.nan
    df['ob_price_bottom'] = np.nan
    
    # Calculate ATR if not present
    if 'atr' not in df.columns:
        from ..indicators.technical import calculate_atr
        df['atr'] = calculate_atr(df)
    
    for i in range(3, len(df)):
        candle_prev = df.iloc[i-1]
        candle_curr = df.iloc[i]
        
        move_size = abs(candle_curr['close'] - candle_prev['close'])
        avg_atr = df.iloc[max(0, i-14):i]['atr'].mean()
        
        # Strong bullish move (displacement)
        if move_size > (avg_atr * atr_mult) and candle_curr['close'] > candle_prev['close']:
            # Last bearish candle before move is the order block
            for j in range(i-1, max(0, i-10), -1):
                if df.iloc[j]['close'] < df.iloc[j]['open']:  # Bearish candle
                    df.loc[df.index[j], 'ob_bullish'] = True
                    df.loc[df.index[j], 'ob_price_top'] = df.iloc[j]['high']
                    df.loc[df.index[j], 'ob_price_bottom'] = df.iloc[j]['low']
                    break
        
        # Strong bearish move (displacement)
        elif move_size > (avg_atr * atr_mult) and candle_curr['close'] < candle_prev['close']:
            # Last bullish candle before move is the order block
            for j in range(i-1, max(0, i-10), -1):
                if df.iloc[j]['close'] > df.iloc[j]['open']:  # Bullish candle
                    df.loc[df.index[j], 'ob_bearish'] = True
                    df.loc[df.index[j], 'ob_price_top'] = df.iloc[j]['high']
                    df.loc[df.index[j], 'ob_price_bottom'] = df.iloc[j]['low']
                    break
    
    return df


def detect_displacement(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Detect displacement candles (strong institutional moves)
    
    Args:
        df: DataFrame with OHLCV data
        atr_mult: ATR multiplier threshold
        
    Returns:
        DataFrame with displacement markers
    """
    df = df.copy()
    df['displacement_bullish'] = False
    df['displacement_bearish'] = False
    
    # Calculate ATR if not present
    if 'atr' not in df.columns:
        from ..indicators.technical import calculate_atr
        df['atr'] = calculate_atr(df)
    
    for i in range(1, len(df)):
        candle = df.iloc[i]
        candle_size = abs(candle['close'] - candle['open'])
        avg_atr = df.iloc[max(0, i-14):i]['atr'].mean()
        
        # Bullish displacement
        if candle_size > (avg_atr * atr_mult) and candle['close'] > candle['open']:
            df.loc[df.index[i], 'displacement_bullish'] = True
        
        # Bearish displacement
        elif candle_size > (avg_atr * atr_mult) and candle['close'] < candle['open']:
            df.loc[df.index[i], 'displacement_bearish'] = True
    
    return df


def apply_all_smc_detectors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply all SMC detection methods
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary with detection parameters
        
    Returns:
        DataFrame with all SMC markers
    """
    logger.info("Applying SMC detectors...")
    
    # Break of Structure
    df = detect_break_of_structure(df, lookback=params.get('bos_lookback', 20))
    
    # Change of Character
    df = detect_change_of_character(df, lookback=params.get('bos_lookback', 20))
    
    # Liquidity Sweeps
    df = detect_liquidity_sweep(
        df, 
        wick_ratio=params.get('liquidity_sweep_wick_ratio', 0.6),
        lookback=10
    )
    
    # Order Blocks
    df = detect_order_blocks(df, atr_mult=params.get('order_block_strength', 1.5))
    
    # Displacement
    df = detect_displacement(df, atr_mult=params.get('order_block_strength', 1.5))
    
    logger.info("SMC detection complete")
    
    return df


def check_smc_confluence(df: pd.DataFrame, idx: int) -> Dict:
    """
    Check for SMC signal confluence at given index
    
    Args:
        df: DataFrame with SMC markers
        idx: Index to check
        
    Returns:
        Dictionary with confluence information
    """
    if idx >= len(df):
        return {}
    
    row = df.iloc[idx]
    
    confluence = {
        'bos_bullish': row.get('bos_bullish', False),
        'bos_bearish': row.get('bos_bearish', False),
        'choch_bullish': row.get('choch_bullish', False),
        'choch_bearish': row.get('choch_bearish', False),
        'sweep_bullish': row.get('sweep_bullish', False),
        'sweep_bearish': row.get('sweep_bearish', False),
        'ob_bullish': row.get('ob_bullish', False),
        'ob_bearish': row.get('ob_bearish', False),
        'displacement_bullish': row.get('displacement_bullish', False),
        'displacement_bearish': row.get('displacement_bearish', False),
    }
    
    # Count bullish and bearish signals
    bullish_count = sum([
        confluence['bos_bullish'],
        confluence['choch_bullish'],
        confluence['sweep_bullish'],
        confluence['ob_bullish'],
        confluence['displacement_bullish']
    ])
    
    bearish_count = sum([
        confluence['bos_bearish'],
        confluence['choch_bearish'],
        confluence['sweep_bearish'],
        confluence['ob_bearish'],
        confluence['displacement_bearish']
    ])
    
    confluence['bullish_count'] = bullish_count
    confluence['bearish_count'] = bearish_count
    confluence['direction'] = 'bullish' if bullish_count > bearish_count else ('bearish' if bearish_count > bullish_count else 'neutral')
    
    return confluence