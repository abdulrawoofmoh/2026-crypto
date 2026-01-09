"""
Trading Strategy
Signal generation and probability scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Main trading strategy with probability scoring"""
    
    def __init__(self, params: dict, scoring_weights: dict, thresholds: dict, penalties: dict):
        """
        Initialize strategy
        
        Args:
            params: Strategy parameters
            scoring_weights: Scoring weights for each factor
            thresholds: Probability thresholds
            penalties: Penalty factors
        """
        self.params = params
        self.weights = scoring_weights
        self.thresholds = thresholds
        self.penalties = penalties
        self.consecutive_losses = 0
        
    def calculate_probability_score(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, 
                                    df_4h: pd.DataFrame, idx: int, 
                                    order_book: Optional[Dict] = None) -> Dict:
        """
        Calculate probability score for a trade setup
        
        Args:
            df_15m: 15-minute DataFrame with all indicators
            df_1h: 1-hour DataFrame
            df_4h: 4-hour DataFrame
            idx: Current index in 15m timeframe
            order_book: Order book data (optional)
            
        Returns:
            Dictionary with score and component breakdown
        """
        if idx >= len(df_15m):
            return {'total_score': 0, 'signal': None}
        
        score = 0
        components = {}
        row = df_15m.iloc[idx]
        
        # Import trend checking
        from .indicators.technical import identify_trend, check_trend_alignment
        
        # 1. TREND ALIGNMENT (25 points max)
        trend_15m = identify_trend(df_15m, idx)
        
        # Get trend alignment across timeframes
        try:
            alignment = check_trend_alignment(df_15m, df_1h, df_4h, idx)
            trend_1h = alignment['trend_1h']
            trend_4h = alignment['trend_4h']
        except:
            trend_1h = 'neutral'
            trend_4h = 'neutral'
        
        if trend_15m == 'bullish' or trend_15m == 'bearish':
            score += self.weights['trend_15m']
            components['trend_15m'] = self.weights['trend_15m']
        
        if trend_1h == trend_15m and trend_1h != 'neutral':
            score += self.weights['trend_1h']
            components['trend_1h'] = self.weights['trend_1h']
        
        if trend_4h == trend_15m and trend_4h != 'neutral':
            score += self.weights['trend_4h']
            components['trend_4h'] = self.weights['trend_4h']
        
        # PENALTY: Against 4H trend
        if trend_4h != 'neutral' and trend_15m != 'neutral' and trend_4h != trend_15m:
            score += self.penalties['against_4h_trend']
            components['against_4h_trend'] = self.penalties['against_4h_trend']
        
        # 2. LIQUIDITY FACTORS (25 points max)
        
        # Order book imbalance
        if order_book:
            from .detectors.liquidity import analyze_order_book_imbalance
            ob_analysis = analyze_order_book_imbalance(order_book, self.params['order_book_imbalance_threshold'])
            
            if (ob_analysis['signal'] == 'bullish' and trend_15m == 'bullish') or \
               (ob_analysis['signal'] == 'bearish' and trend_15m == 'bearish'):
                score += self.weights['order_book_imbalance']
                components['order_book_imbalance'] = self.weights['order_book_imbalance']
        
        # Near volume node (POC/VAH/VAL)
        # This would be checked in signal generation
        
        # Near order block
        if row.get('ob_bullish', False) and trend_15m == 'bullish':
            score += self.weights['order_block']
            components['order_block'] = self.weights['order_block']
        elif row.get('ob_bearish', False) and trend_15m == 'bearish':
            score += self.weights['order_block']
            components['order_block'] = self.weights['order_block']
        
        # Fair Value Gap
        if row.get('fvg_bullish', False) and trend_15m == 'bullish':
            score += self.weights['fair_value_gap']
            components['fair_value_gap'] = self.weights['fair_value_gap']
        elif row.get('fvg_bearish', False) and trend_15m == 'bearish':
            score += self.weights['fair_value_gap']
            components['fair_value_gap'] = self.weights['fair_value_gap']
        
        # 3. SMC SIGNALS (25 points max)
        
        # Break of Structure
        if row.get('bos_bullish', False) and trend_15m == 'bullish':
            score += self.weights['break_of_structure']
            components['break_of_structure'] = self.weights['break_of_structure']
        elif row.get('bos_bearish', False) and trend_15m == 'bearish':
            score += self.weights['break_of_structure']
            components['break_of_structure'] = self.weights['break_of_structure']
        
        # Liquidity Sweep (VERY STRONG signal)
        if row.get('sweep_bullish', False):
            score += self.weights['liquidity_sweep']
            components['liquidity_sweep'] = self.weights['liquidity_sweep']
        elif row.get('sweep_bearish', False):
            score += self.weights['liquidity_sweep']
            components['liquidity_sweep'] = self.weights['liquidity_sweep']
        
        # Displacement
        if row.get('displacement_bullish', False) and trend_15m == 'bullish':
            score += self.weights['displacement']
            components['displacement'] = self.weights['displacement']
        elif row.get('displacement_bearish', False) and trend_15m == 'bearish':
            score += self.weights['displacement']
            components['displacement'] = self.weights['displacement']
        
        # 4. TECHNICAL CONFLUENCE (15 points max)
        
        # RSI in optimal zone
        rsi = row.get('rsi', 50)
        if trend_15m == 'bullish' and self.params['rsi_long_zone'][0] < rsi < self.params['rsi_long_zone'][1]:
            score += self.weights['rsi_zone']
            components['rsi_zone'] = self.weights['rsi_zone']
        elif trend_15m == 'bearish' and self.params['rsi_short_zone'][0] < rsi < self.params['rsi_short_zone'][1]:
            score += self.weights['rsi_zone']
            components['rsi_zone'] = self.weights['rsi_zone']
        
        # Volume surge
        volume_ratio = row.get('volume_ratio', 1.0)
        if volume_ratio >= self.params['volume_threshold']:
            score += self.weights['volume_surge']
            components['volume_surge'] = self.weights['volume_surge']
        else:
            # PENALTY: Low volume
            score += self.penalties['low_volume']
            components['low_volume'] = self.penalties['low_volume']
        
        # RSI Divergence
        if row.get('rsi_divergence', 0) != 0:
            score += self.weights['rsi_divergence']
            components['rsi_divergence'] = self.weights['rsi_divergence']
        
        # 5. ENTRY QUALITY (10 points max)
        # These would be checked during signal generation
        
        # 6. PENALTIES
        
        # Recent losing streak
        if self.consecutive_losses >= 3:
            score += self.penalties['recent_losses']
            components['recent_losses'] = self.penalties['recent_losses']
        
        # Determine signal direction
        signal = None
        if score >= self.thresholds['minimum_score']:
            if trend_15m == 'bullish':
                signal = 'LONG'
            elif trend_15m == 'bearish':
                signal = 'SHORT'
        
        return {
            'total_score': score,
            'signal': signal,
            'components': components,
            'trend_15m': trend_15m,
            'trend_1h': trend_1h,
            'trend_4h': trend_4h
        }
    
    def generate_signal(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, 
                       df_4h: pd.DataFrame, idx: int,
                       volume_profile: Dict = None,
                       order_book: Dict = None) -> Optional[Dict]:
        """
        Generate trading signal
        
        Args:
            df_15m: 15-minute DataFrame
            df_1h: 1-hour DataFrame
            df_4h: 4-hour DataFrame
            idx: Current index
            volume_profile: Volume profile data
            order_book: Order book data
            
        Returns:
            Signal dictionary or None
        """
        # Calculate probability score
        score_result = self.calculate_probability_score(df_15m, df_1h, df_4h, idx, order_book)
        
        if score_result['signal'] is None:
            return None
        
        row = df_15m.iloc[idx]
        
        # Additional filters
        
        # Check trend strength
        if row.get('trend_strength', 0) < self.params['trend_strength_min']:
            return None
        
        # Check if near volume profile levels (support for LONG, resistance for SHORT)
        if volume_profile:
            current_price = row['close']
            poc = volume_profile.get('poc', 0)
            
            # Near POC is good for entries
            distance_to_poc = abs(current_price - poc) / current_price
            if distance_to_poc < 0.01:  # Within 1% of POC
                score_result['total_score'] += self.weights['volume_node']
        
        # Final check
        if score_result['total_score'] < self.thresholds['minimum_score']:
            return None
        
        # Calculate stops and targets
        atr = row.get('atr', 0)
        entry_price = row['close']
        
        if score_result['signal'] == 'LONG':
            stop_loss = entry_price - (atr * self.params['sl_atr_mult'])
            take_profit = entry_price + (atr * self.params['sl_atr_mult'] * self.params['tp_ratio'])
        else:  # SHORT
            stop_loss = entry_price + (atr * self.params['sl_atr_mult'])
            take_profit = entry_price - (atr * self.params['sl_atr_mult'] * self.params['tp_ratio'])
        
        return {
            'signal': score_result['signal'],
            'score': score_result['total_score'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': row['timestamp'],
            'components': score_result['components'],
            'confidence': 'elite' if score_result['total_score'] >= self.thresholds['elite_signal'] else \
                         'strong' if score_result['total_score'] >= self.thresholds['strong_signal'] else 'minimum'
        }
    
    def update_loss_streak(self, is_win: bool):
        """Update consecutive loss counter"""
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1