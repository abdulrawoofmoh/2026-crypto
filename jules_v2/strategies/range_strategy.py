"""Range Trading Strategy - Enhanced Short Grid"""

import pandas as pd
import numpy as np

class RangeStrategy:

    def __init__(self, params):
        self.params = params

    def generate_signal(self, df, idx):
        if idx < 50: return None
        row = df.iloc[idx]

        # Check ADX (must be ranging)
        if row['adx'] > self.params['adx_threshold']:
            return None

        signal_type = None

        # LONG: Oversold
        if row['rsi'] < self.params['rsi_oversold']:
            if row['close'] < row['bb_lower']:
                if row['close'] > df.iloc[idx-1]['close']:
                    signal_type = 'LONG'

        # SHORT: Overbought (Aggressive for Bear Market)
        if row['rsi'] > self.params['rsi_overbought']:
            if row['close'] > row['bb_upper']:
                if row['close'] < df.iloc[idx-1]['close']:
                    signal_type = 'SHORT'

        if not signal_type:
            return None

        # Confidence Score
        confidence = 0.8 # Boost base confidence
        return {'type': signal_type, 'confidence': confidence}

    def calculate_stops(self, df, idx, signal_dict):
        row = df.iloc[idx]
        atr = row['atr']
        entry_price = row['close']
        signal = signal_dict['type']

        # WIDER STOPS FOR GRID SURVIVAL
        stop_distance = atr * 5.0 # Very wide to allow grid to work

        if signal == 'LONG':
            stop_loss = entry_price - stop_distance
            # Tight profit for churning
            take_profit = entry_price + (atr * 0.5)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (atr * 0.5)

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'use_trailing': False
        }

    def check_scale_in(self, df, idx, position):
        if not self.params.get('scale_in_enabled', False): return None
        if position['scale_ins'] >= self.params.get('max_scale_steps', 0): return None

        row = df.iloc[idx]
        atr = row['atr']
        step = self.params.get('scale_step_atr', 1.0) * atr

        if position['type'] == 'LONG':
            if row['close'] < position['entry_price'] - step:
                return 'LONG'
        elif position['type'] == 'SHORT':
            if row['close'] > position['entry_price'] + step:
                return 'SHORT'
        return None
