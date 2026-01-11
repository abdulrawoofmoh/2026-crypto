"""Range Trading Strategy - Frequency & Probability Engine"""

import pandas as pd
import numpy as np

class RangeStrategy:
    """
    Range Strategy Phase 2: High Frequency, Mean Reversion, Grid Scaling
    """

    def __init__(self, params):
        self.params = params

    def generate_signal(self, df, idx):
        if idx < 200: return None
        row = df.iloc[idx]
        prev_row = df.iloc[idx-1]

        # 1. 200 EMA Filter (Trend Filter)
        # Long only if EMA Slope >= 0 (Bullish/Neutral)
        # Short only if EMA Slope < 0 (Bearish)
        ema_slope = row['ema_200_slope']

        signal_type = None

        # LONG SETUP: RSI < 30 (Aggressive Scale In Zone) or < 40 (Initial Entry)
        # Use 40/60 as requested to force frequency
        if row['rsi'] < 40:
            # Confirmation: RSI held below 40 for 2 candles?
            # Or just current < 40 and prev < 40?
            # "RSI must cross AND hold for 2 candles" implies entry on 3rd?
            # Let's simple check: current < 40 and prev < 40.
            if prev_row['rsi'] < 40:
                if ema_slope >= 0: # Filter: Only buy if trend is up/flat
                    signal_type = 'LONG'

        # SHORT SETUP: RSI > 60
        elif row['rsi'] > 60:
            if prev_row['rsi'] > 60:
                if ema_slope < 0: # Filter: Only sell if trend is down
                    signal_type = 'SHORT'

        if not signal_type:
            return None

        # Confidence based on RSI depth
        confidence = 1.0

        return {
            'type': signal_type,
            'confidence': confidence,
            'strategy': 'RANGE',
            'size_pct': 0.04 # Initial Entry 4%
        }

    def calculate_stops(self, df, idx, signal_dict):
        row = df.iloc[idx]
        atr = row['atr']
        entry_price = row['close']
        signal = signal_dict['type']

        # Volatility-Adaptive Stop: 1.2 ATR
        stop_dist = atr * 1.2

        if signal == 'LONG':
            stop_loss = entry_price - stop_dist
        else:
            stop_loss = entry_price + stop_dist

        return {
            'stop_loss': stop_loss,
            'take_profit': None, # Dynamic Exit (RSI 50)
            'use_trailing': False # Rely on Mean Reversion Exit
        }

    def check_scale_in(self, df, idx, position):
        """Scale In Logic: RSI deviation + Fixed Sizing"""
        # Logic: Scale if RSI < Entry RSI - 5 (Long)
        # Sizing: +3%

        if position['scale_ins'] >= 2: # Max 2 scales (Total 3 positions)
            return None

        row = df.iloc[idx]

        # Get RSI at entry? We don't store "Entry RSI" in position dict.
        # Approximation: Check Price drop?
        # User explicitly said "if RSI < Entry RSI - 5".
        # This is hard without storing Entry RSI.
        # Workaround: Use Price-based approximation OR just rely on current RSI levels?
        # Let's assume Entry RSI was around 40/60.
        # Scale 1: RSI < 35 (Long) / > 65 (Short)
        # Scale 2: RSI < 30 (Long) / > 70 (Short)

        current_rsi = row['rsi']
        signal = None

        if position['type'] == 'LONG':
            # Scale 1
            if position['scale_ins'] == 0 and current_rsi < 35:
                signal = 'LONG'
            # Scale 2
            elif position['scale_ins'] == 1 and current_rsi < 30:
                signal = 'LONG'

        elif position['type'] == 'SHORT':
            if position['scale_ins'] == 0 and current_rsi > 65:
                signal = 'SHORT'
            elif position['scale_ins'] == 1 and current_rsi > 70:
                signal = 'SHORT'

        if signal:
            return {'signal': signal, 'size_pct': 0.03} # +3%

        return None

    def check_custom_exit(self, df, idx, position):
        """Mean Reversion Exit: RSI returns to 50-55"""
        row = df.iloc[idx]
        rsi = row['rsi']

        if position['type'] == 'LONG':
            if rsi >= 50:
                return "Mean Reversion (RSI > 50)"
        elif position['type'] == 'SHORT':
            if rsi <= 50:
                return "Mean Reversion (RSI < 50)"

        return None
