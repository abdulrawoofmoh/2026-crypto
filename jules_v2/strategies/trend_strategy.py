"""Trend Following Strategy - Liquidity Breakout based with Probability Scoring"""

import pandas as pd
import numpy as np

class TrendStrategy:
    """
    Trend Strategy V2: Liquidity Breakouts with Confidence Scoring
    """

    def __init__(self, params):
        self.params = params

    def generate_signal(self, df, idx):
        """
        Generate trading signal with confidence score.
        Returns: dict {'type': 'LONG'/'SHORT', 'confidence': float (0.5 to 1.0)} or None
        """
        if idx < 50:
            return None

        row = df.iloc[idx]

        # 1. Identify Liquidity Levels
        lookback = self.params['lookback_period']
        recent_high = df.iloc[idx-lookback:idx]['high'].max()
        recent_low = df.iloc[idx-lookback:idx]['low'].min()

        # 2. Volume Check
        vol_ratio = row['volume_ratio']
        if vol_ratio < self.params['volume_multiplier']:
            return None

        signal_type = None

        # 3. Breakout Detection
        if row['close'] > recent_high:
            if 50 < row['rsi'] < 80: # Healthy Bullish RSI
                 if row['adx'] > self.params['adx_threshold']:
                     signal_type = 'LONG'

        elif row['close'] < recent_low:
            if 20 < row['rsi'] < 50: # Healthy Bearish RSI
                 if row['adx'] > self.params['adx_threshold']:
                     signal_type = 'SHORT'

        if not signal_type:
            return None

        # 4. Calculate Confidence Score (0.5 - 1.0)
        # Factors:
        # - Volume: Higher is better (capped at 3.0x ratio)
        # - ADX: Higher is better (capped at 50)
        # - Breakout Strength: (Close - Level) / Level

        # Volume Score (0.0 to 0.4)
        v_score = min((vol_ratio - self.params['volume_multiplier']) / 2.0, 0.4)
        if v_score < 0: v_score = 0

        # ADX Score (0.0 to 0.3)
        a_score = min((row['adx'] - self.params['adx_threshold']) / 20.0, 0.3)
        if a_score < 0: a_score = 0

        # Base confidence 0.5 + bonuses
        confidence = 0.5 + v_score + a_score

        # Cap at 1.0
        confidence = min(confidence, 1.0)

        return {
            'type': signal_type,
            'confidence': confidence
        }

    def calculate_stops(self, df, idx, signal_dict):
        """Calculate stops"""
        row = df.iloc[idx]
        atr = row['atr']
        entry_price = row['close']
        signal = signal_dict['type']

        if signal == 'LONG':
            stop_loss = entry_price - (atr * self.params['stop_loss_atr'])
        else:
            stop_loss = entry_price + (atr * self.params['stop_loss_atr'])

        return {
            'stop_loss': stop_loss,
            'take_profit': None,
            'use_trailing': False
        }
