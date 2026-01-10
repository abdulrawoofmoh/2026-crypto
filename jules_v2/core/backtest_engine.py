"""Backtesting engine - CRITICAL: Accurate P&L calculation"""

import pandas as pd
import numpy as np
import logging
from config.settings import RISK_PARAMS

logger = logging.getLogger(__name__)

class BacktestEngine:

    def __init__(self, initial_balance, leverage, base_position_size, commission, slippage, profit_retracement_pct=None):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.base_position_size_pct = base_position_size
        self.commission = commission
        self.slippage = slippage
        self.profit_retracement_pct = profit_retracement_pct if profit_retracement_pct is not None else 0.25

        self.position = None
        self.trades = []
        self.equity_curve = [initial_balance]

    def open_position(self, signal_data, entry_price, stop_loss, take_profit, use_trailing=False, trailing_pct=0.8, timestamp=None):
        if self.position is not None or self.balance <= 0: return

        signal_type = signal_data['type']
        confidence = signal_data.get('confidence', 1.0)

        if signal_type == 'LONG':
            actual_entry = entry_price * (1 + self.slippage)
        else:
            actual_entry = entry_price * (1 - self.slippage)

        alloc_pct = self.base_position_size_pct * confidence
        margin_used = self.balance * alloc_pct
        position_value = margin_used * self.leverage

        if position_value < 10: return

        position_size_units = position_value / actual_entry
        commission_cost = position_value * self.commission
        self.balance -= commission_cost

        self.position = {
            'type': signal_type,
            'entry_price': actual_entry,
            'size': position_size_units,
            'margin': margin_used,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'use_trailing': use_trailing,
            'profit_retracement_pct': self.profit_retracement_pct,
            'highest_unrealized_pnl': 0.0,
            'scale_ins': 0,
            'confidence_at_entry': confidence
        }

    def scale_position(self, current_price, signal_type, timestamp):
        if self.position is None: return
        if self.position['type'] != signal_type: return

        if signal_type == 'LONG':
            actual_entry = current_price * (1 + self.slippage)
        else:
            actual_entry = current_price * (1 - self.slippage)

        alloc_pct = self.base_position_size_pct * self.position['confidence_at_entry']
        margin_used = self.balance * alloc_pct

        if margin_used > self.balance * 0.9: return

        position_value = margin_used * self.leverage
        position_size_units = position_value / actual_entry
        commission_cost = position_value * self.commission
        self.balance -= commission_cost

        total_units = self.position['size'] + position_size_units
        total_value = (self.position['size'] * self.position['entry_price']) + (position_size_units * actual_entry)
        avg_entry = total_value / total_units

        self.position['entry_price'] = avg_entry
        self.position['size'] = total_units
        self.position['margin'] += margin_used
        self.position['scale_ins'] += 1

    def close_position(self, exit_price, reason, timestamp):
        if self.position is None: return

        if self.position['type'] == 'LONG':
            actual_exit = exit_price * (1 - self.slippage)
            price_change_pct = (actual_exit - self.position['entry_price']) / self.position['entry_price']
        else:
            actual_exit = exit_price * (1 + self.slippage)
            price_change_pct = (self.position['entry_price'] - actual_exit) / self.position['entry_price']

        pnl_pct = price_change_pct * self.leverage * 100
        pnl_usd = self.position['margin'] * price_change_pct * self.leverage

        position_value = self.position['size'] * actual_exit
        commission_cost = position_value * self.commission
        pnl_usd -= commission_cost

        self.balance += pnl_usd
        if self.balance < 0: self.balance = 0

        self.trades.append({
            'entry_price': self.position['entry_price'],
            'exit_price': actual_exit,
            'type': self.position['type'],
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'balance': self.balance,
            'exit_reason': reason,
            'timestamp': timestamp,
            'scale_ins': self.position['scale_ins']
        })

        self.position = None

    def check_profit_trailing(self, current_price, timestamp):
        if not self.position: return False

        if self.position['type'] == 'LONG':
            diff = current_price - self.position['entry_price']
            pnl_val = diff * self.position['size']
        else:
            diff = self.position['entry_price'] - current_price
            pnl_val = diff * self.position['size']

        if pnl_val <= 0: return False

        if pnl_val > self.position['highest_unrealized_pnl']:
            self.position['highest_unrealized_pnl'] = pnl_val

        trailing_threshold = self.position['highest_unrealized_pnl'] * (1 - self.position['profit_retracement_pct'])

        if pnl_val < trailing_threshold:
            self.close_position(current_price, 'Profit Retracement Trailing', timestamp)
            return True
        return False

    def check_exits(self, row):
        if not self.position: return

        current_price = row['close']
        high = row['high']
        low = row['low']
        timestamp = row['timestamp']

        if self.position['type'] == 'LONG':
            if low <= self.position['stop_loss']:
                self.close_position(self.position['stop_loss'], 'Stop Loss', timestamp)
                return
        else:
            if high >= self.position['stop_loss']:
                self.close_position(self.position['stop_loss'], 'Stop Loss', timestamp)
                return

        if self.check_profit_trailing(current_price, timestamp):
            return

        if self.position['take_profit']:
            if self.position['type'] == 'LONG' and high >= self.position['take_profit']:
                 self.close_position(self.position['take_profit'], 'Take Profit', timestamp)
                 return
            elif self.position['type'] == 'SHORT' and low <= self.position['take_profit']:
                 self.close_position(self.position['take_profit'], 'Take Profit', timestamp)
                 return

    def run(self, df, strategy):
        for idx in range(len(df)):
            row = df.iloc[idx]
            self.check_exits(row)

            if self.position is None and self.balance > 0:
                signal_data = strategy.generate_signal(df, idx)
                if signal_data:
                    stops = strategy.calculate_stops(df, idx, signal_data)
                    self.open_position(
                        signal_data, row['close'], stops['stop_loss'],
                        stops.get('take_profit'), False,
                        0.0, row['timestamp']
                    )
            elif self.position and self.balance > 0:
                if hasattr(strategy, 'check_scale_in'):
                    scale_signal = strategy.check_scale_in(df, idx, self.position)
                    if scale_signal:
                        self.scale_position(row['close'], scale_signal, row['timestamp'])

            current_equity = self.balance
            if self.position:
                current_price = row['close']
                if self.position['type'] == 'LONG':
                    unrealized_pct = (current_price - self.position['entry_price']) / self.position['entry_price']
                else:
                    unrealized_pct = (self.position['entry_price'] - current_price) / self.position['entry_price']
                unrealized_pnl = self.position['margin'] * unrealized_pct * self.leverage
                current_equity += unrealized_pnl

            self.equity_curve.append(current_equity)

        if self.position:
            self.close_position(df.iloc[-1]['close'], 'End of Backtest', df.iloc[-1]['timestamp'])

        return self.calculate_stats()

    def calculate_stats(self):
        """Calculate stats and return robust dictionary"""
        if not self.trades:
            return {
                'total_trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0, 'total_return': 0, 'max_drawdown': 0,
                'sharpe_ratio': 0, 'final_balance': self.balance,
                'avg_win': 0, 'avg_loss': 0, 'trades': []
            }

        trades_df = pd.DataFrame(self.trades)
        wins = len(trades_df[trades_df['pnl_usd'] > 0])
        losses = len(trades_df[trades_df['pnl_usd'] <= 0])
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100

        equity_series = pd.Series(self.equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        returns = trades_df['pnl_pct'].values
        sharpe_ratio = 0
        if len(returns) > 1 and returns.std() > 0:
             sharpe_ratio = returns.mean() / returns.std()

        avg_win = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl_usd'] <= 0]['pnl_usd'].mean() if losses > 0 else 0

        return {
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': (wins/len(self.trades))*100,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': self.trades
        }
