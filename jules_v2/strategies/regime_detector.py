"""Strategy Manager - Adjusted for Bear/Short Detection"""

import pandas as pd
import numpy as np

class StrategyManager:
    """
    Decides which strategy (Trend or Range) to deploy based on market conditions.
    """

    def __init__(self, params):
        self.params = params

    def detect_regime(self, df, idx):
        if idx < 50:
            return 'UNCERTAIN'

        row = df.iloc[idx]
        adx = row['adx']

        # Calculate price distance from Slow MA
        price_ma_distance = (row['close'] - row['ma_slow']) / row['ma_slow'] * 100

        # 1. Strong Trend (Bull or Bear)
        if adx > self.params['trend_threshold']:
            # If price is far from MA, it's trending
            if abs(price_ma_distance) > self.params['ma_distance_pct']:
                return 'TRENDING'

        # 2. Ranging
        if adx < self.params['range_threshold']:
            return 'RANGING'

        # 3. Default to RANGING if uncertain (to enable Grid)
        return 'RANGING'

    def add_regime_column(self, df):
        df['regime'] = 'UNCERTAIN'
        regimes = []
        for idx in range(len(df)):
            regimes.append(self.detect_regime(df, idx))
        df['regime'] = regimes
        return df
