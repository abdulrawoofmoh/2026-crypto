"""
Liquidity Detection
Volume profile, order book analysis, swing clusters, Fair Value Gaps
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_volume_profile(df: pd.DataFrame, bins: int = 50) -> Dict:
    """
    Calculate volume profile to find high-volume price levels
    
    Args:
        df: DataFrame with OHLCV data
        bins: Number of price bins
        
    Returns:
        Dictionary with POC, VAH, VAL, and volume distribution
    """
    if len(df) < 10:
        return {}
    
    # Get price range
    price_min = df['low'].min()
    price_max = df['high'].max()
    
    # Create price bins
    price_bins = np.linspace(price_min, price_max, bins)
    
    # Calculate volume at each price level
    volume_at_price = np.zeros(bins - 1)
    
    for i in range(len(df)):
        candle_low = df.iloc[i]['low']
        candle_high = df.iloc[i]['high']
        candle_volume = df.iloc[i]['volume']
        
        # Distribute volume across price bins this candle touched
        for j in range(len(price_bins) - 1):
            bin_low = price_bins[j]
            bin_high = price_bins[j + 1]
            
            # Check if candle overlaps with this bin
            if candle_high >= bin_low and candle_low <= bin_high:
                # Add proportional volume
                overlap = min(candle_high, bin_high) - max(candle_low, bin_low)
                candle_range = candle_high - candle_low
                if candle_range > 0:
                    volume_at_price[j] += candle_volume * (overlap / candle_range)
    
    # Find Point of Control (POC) - price with highest volume
    poc_idx = np.argmax(volume_at_price)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
    
    # Find Value Area (70% of volume)
    total_volume = volume_at_price.sum()
    value_area_volume = total_volume * 0.70
    
    # Start from POC and expand until we have 70% of volume
    current_volume = volume_at_price[poc_idx]
    lower_idx = poc_idx
    upper_idx = poc_idx
    
    while current_volume < value_area_volume:
        # Expand to side with more volume
        lower_vol = volume_at_price[lower_idx - 1] if lower_idx > 0 else 0
        upper_vol = volume_at_price[upper_idx + 1] if upper_idx < len(volume_at_price) - 1 else 0
        
        if lower_vol > upper_vol and lower_idx > 0:
            lower_idx -= 1
            current_volume += lower_vol
        elif upper_idx < len(volume_at_price) - 1:
            upper_idx += 1
            current_volume += upper_vol
        else:
            break
    
    vah_price = price_bins[upper_idx + 1]  # Value Area High
    val_price = price_bins[lower_idx]      # Value Area Low
    
    # Find high volume nodes (top 20%)
    volume_threshold = np.percentile(volume_at_price, 80)
    high_volume_nodes = []
    
    for i in range(len(volume_at_price)):
        if volume_at_price[i] >= volume_threshold:
            node_price = (price_bins[i] + price_bins[i + 1]) / 2
            high_volume_nodes.append({
                'price': node_price,
                'volume': volume_at_price[i]
            })
    
    return {
        'poc': poc_price,
        'vah': vah_price,
        'val': val_price,
        'high_volume_nodes': high_volume_nodes,
        'volume_distribution': volume_at_price,
        'price_bins': price_bins
    }


def analyze_order_book_imbalance(order_book: Dict, threshold: float = 1.5) -> Dict:
    """
    Analyze order book for bid/ask imbalance
    
    Args:
        order_book: Order book data from exchange
        threshold: Imbalance threshold
        
    Returns:
        Dictionary with imbalance analysis
    """
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return {'signal': 'neutral', 'ratio': 1.0}
    
    bid_volume = order_book.get('bid_volume', 0)
    ask_volume = order_book.get('ask_volume', 0)
    imbalance_ratio = order_book.get('imbalance_ratio', 1.0)
    
    # Determine signal
    if imbalance_ratio > threshold:
        signal = 'bullish'
    elif imbalance_ratio < (1 / threshold):
        signal = 'bearish'
    else:
        signal = 'neutral'
    
    return {
        'signal': signal,
        'ratio': imbalance_ratio,
        'bid_volume': bid_volume,
        'ask_volume': ask_volume
    }


def find_large_order_walls(order_book: Dict, multiplier: float = 5.0) -> List[Dict]:
    """
    Find large order walls in order book
    
    Args:
        order_book: Order book data
        multiplier: Size multiplier vs average
        
    Returns:
        List of large order walls
    """
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return []
    
    walls = []
    
    # Analyze bids
    bids = order_book['bids']
    if bids:
        bid_sizes = [bid[1] for bid in bids]
        avg_bid_size = np.mean(bid_sizes)
        threshold = avg_bid_size * multiplier
        
        for bid in bids:
            price, size = bid[0], bid[1]
            if size >= threshold:
                walls.append({
                    'price': price,
                    'size': size,
                    'side': 'bid',
                    'strength': size / avg_bid_size
                })
    
    # Analyze asks
    asks = order_book['asks']
    if asks:
        ask_sizes = [ask[1] for ask in asks]
        avg_ask_size = np.mean(ask_sizes)
        threshold = avg_ask_size * multiplier
        
        for ask in asks:
            price, size = ask[0], ask[1]
            if size >= threshold:
                walls.append({
                    'price': price,
                    'size': size,
                    'side': 'ask',
                    'strength': size / avg_ask_size
                })
    
    return walls


def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Detect swing highs and lows
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of candles to look back
        
    Returns:
        DataFrame with swing high/low markers
    """
    df = df.copy()
    df['swing_high'] = False
    df['swing_low'] = False
    
    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if df.iloc[i]['high'] <= df.iloc[i - j]['high'] or df.iloc[i]['high'] <= df.iloc[i + j]['high']:
                is_swing_high = False
                break
        df.loc[df.index[i], 'swing_high'] = is_swing_high
        
        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if df.iloc[i]['low'] >= df.iloc[i - j]['low'] or df.iloc[i]['low'] >= df.iloc[i + j]['low']:
                is_swing_low = False
                break
        df.loc[df.index[i], 'swing_low'] = is_swing_low
    
    return df


def find_swing_clusters(df: pd.DataFrame, distance_pct: float = 0.02) -> List[Dict]:
    """
    Find clusters of swing points (areas with multiple swing highs/lows)
    
    Args:
        df: DataFrame with swing points detected
        distance_pct: Maximum distance between swings (as percentage)
        
    Returns:
        List of swing clusters
    """
    if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
        df = detect_swing_points(df)
    
    clusters = []
    
    # Find swing high clusters
    swing_highs = df[df['swing_high'] == True]
    for i, swing1 in swing_highs.iterrows():
        cluster_swings = [swing1['high']]
        
        for j, swing2 in swing_highs.iterrows():
            if i != j:
                price_diff = abs(swing1['high'] - swing2['high']) / swing1['high']
                if price_diff <= distance_pct:
                    cluster_swings.append(swing2['high'])
        
        if len(cluster_swings) >= 3:  # At least 3 swings
            clusters.append({
                'type': 'resistance',
                'price': np.mean(cluster_swings),
                'count': len(cluster_swings),
                'strength': len(cluster_swings)
            })
    
    # Find swing low clusters
    swing_lows = df[df['swing_low'] == True]
    for i, swing1 in swing_lows.iterrows():
        cluster_swings = [swing1['low']]
        
        for j, swing2 in swing_lows.iterrows():
            if i != j:
                price_diff = abs(swing1['low'] - swing2['low']) / swing1['low']
                if price_diff <= distance_pct:
                    cluster_swings.append(swing2['low'])
        
        if len(cluster_swings) >= 3:
            clusters.append({
                'type': 'support',
                'price': np.mean(cluster_swings),
                'count': len(cluster_swings),
                'strength': len(cluster_swings)
            })
    
    # Remove duplicates
    unique_clusters = []
    for cluster in clusters:
        is_duplicate = False
        for existing in unique_clusters:
            if existing['type'] == cluster['type']:
                price_diff = abs(cluster['price'] - existing['price']) / cluster['price']
                if price_diff <= distance_pct:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_clusters.append(cluster)
    
    return unique_clusters


def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG) - 3-candle imbalance patterns
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with FVG markers
    """
    df = df.copy()
    df['fvg_bullish'] = False
    df['fvg_bearish'] = False
    df['fvg_price'] = np.nan
    
    for i in range(2, len(df)):
        candle_1 = df.iloc[i - 2]
        candle_2 = df.iloc[i - 1]
        candle_3 = df.iloc[i]
        
        # Bullish FVG: Gap between candle 1 low and candle 3 high
        if candle_1['low'] > candle_3['high']:
            df.loc[df.index[i], 'fvg_bearish'] = True
            df.loc[df.index[i], 'fvg_price'] = (candle_1['low'] + candle_3['high']) / 2
        
        # Bearish FVG: Gap between candle 1 high and candle 3 low  
        elif candle_1['high'] < candle_3['low']:
            df.loc[df.index[i], 'fvg_bullish'] = True
            df.loc[df.index[i], 'fvg_price'] = (candle_1['high'] + candle_3['low']) / 2
    
    return df


def is_near_liquidity_level(current_price: float, liquidity_levels: List[float], tolerance_pct: float = 0.01) -> Tuple[bool, Optional[float]]:
    """
    Check if current price is near a liquidity level
    
    Args:
        current_price: Current market price
        liquidity_levels: List of liquidity price levels
        tolerance_pct: Distance tolerance as percentage
        
    Returns:
        Tuple of (is_near, nearest_level)
    """
    if not liquidity_levels:
        return False, None
    
    for level in liquidity_levels:
        distance_pct = abs(current_price - level) / current_price
        if distance_pct <= tolerance_pct:
            return True, level
    
    return False, None