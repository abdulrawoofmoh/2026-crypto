"""Golden Trio Trend Strategy"""

import pandas as pd
import numpy as np

class GoldenTrioStrategy:
    """
    Golden Trio Strategy
    1. Filter: 200 EMA
    2. Trigger: Supertrend Flip
    3. Confirmation: ADX > 25
    4. Exit: Chandelier Exit (ATR Trailing)
    """

    def __init__(self, params):
        self.params = params

    def generate_signal(self, df, idx):
        """Generate trading signal with confidence score."""
        if idx < 200:
            return None

        row = df.iloc[idx]
        prev_row = df.iloc[idx-1]

        # 1. Filter: 200 EMA
        if row['ema_200'] is None or np.isnan(row['ema_200']):
            return None

        long_term_trend = 'BULL' if row['close'] > row['ema_200'] else 'BEAR'

        # 2. Confirmation: ADX > 25
        if row['adx'] < self.params.get('adx_threshold', 25):
            return None

        signal_type = None

        # 3. Trigger: Supertrend Flip
        # Check if Supertrend Direction changed from prev candle
        # prev_row['supertrend_direction'] vs row['supertrend_direction']

        # LONG SETUP
        if long_term_trend == 'BULL':
            # Flip from Bearish (-1) to Bullish (1)
            if prev_row['supertrend_direction'] == -1 and row['supertrend_direction'] == 1:
                signal_type = 'LONG'

        # SHORT SETUP (Optional in Golden Trio, but good for algos)
        elif long_term_trend == 'BEAR':
            # Flip from Bullish (1) to Bearish (-1)
            if prev_row['supertrend_direction'] == 1 and row['supertrend_direction'] == -1:
                signal_type = 'SHORT'

        if not signal_type:
            return None

        # Confidence Score (0.5 - 1.0) based on Trend Strength
        # ADX 25 -> 0.5, ADX 50 -> 1.0
        adx_score = min((row['adx'] - 25) / 50.0, 0.5)
        confidence = 0.5 + max(0, adx_score)

        return {
            'type': signal_type,
            'confidence': confidence
        }

    def calculate_stops(self, df, idx, signal_dict):
        """Calculate initial stops (Chandelier Exit logic handled by Engine)"""
        # We need to provide a hard Stop Loss for safety,
        # but the main exit is the Chandelier Trailing Stop.
        # Chandelier Exit usually implies a trailing stop from the start.

        row = df.iloc[idx]
        atr = row['atr']
        entry_price = row['close']
        signal = signal_dict['type']

        # Initial Hard Stop (Safety Net)
        # Chandelier is usually High - 3*ATR.
        # So we set initial stop roughly there.
        stop_mult = self.params.get('chandelier_mult', 3.0)

        if signal == 'LONG':
            stop_loss = entry_price - (atr * stop_mult)
        else:
            stop_loss = entry_price + (atr * stop_mult)

        return {
            'stop_loss': stop_loss,
            'take_profit': None, # Let winners run
            'use_trailing': True,
            'trailing_type': 'atr',
            'trailing_val': stop_mult
        }
