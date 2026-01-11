"""Backtesting engine - Refactored for Accuracy, Stability, and Dynamic Exits"""

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
        self.profit_retracement_pct = profit_retracement_pct

        self.position = None
        self.trades = []
        self.equity_curve = [initial_balance]

    def open_position(self, signal_data, entry_price, stop_loss, take_profit, use_trailing=False, trailing_type='percent', trailing_val=0.0, timestamp=None):
        if self.position is not None or self.balance <= 0: return

        signal_type = signal_data['type']
        strategy_name = signal_data.get('strategy', 'UNKNOWN')
        confidence = signal_data.get('confidence', 1.0)

        # Explicit Size override from strategy? (e.g. 4%)
        # Or use base * confidence
        size_override = signal_data.get('size_pct')

        if signal_type == 'LONG':
            actual_entry = entry_price * (1 + self.slippage)
        else:
            actual_entry = entry_price * (1 - self.slippage)

        if size_override:
            alloc_pct = size_override
        else:
            alloc_pct = self.base_position_size_pct * confidence

        margin_used = self.balance * alloc_pct

        if margin_used < 10: return

        notional_value = margin_used * self.leverage
        position_size_units = notional_value / actual_entry

        commission_cost = margin_used * self.commission
        self.balance -= commission_cost

        self.position = {
            'type': signal_type,
            'strategy': strategy_name,
            'entry_price': actual_entry,
            'size': position_size_units,
            'margin': margin_used,
            'leverage': self.leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'use_trailing': use_trailing,
            'trailing_type': trailing_type,
            'trailing_val': trailing_val,
            'profit_retracement_pct': self.profit_retracement_pct,
            'highest_unrealized_pnl': 0.0,
            'highest_price': actual_entry if signal_type == 'LONG' else 0,
            'lowest_price': actual_entry if signal_type == 'SHORT' else float('inf'),
            'scale_ins': 0,
            'confidence_at_entry': confidence
        }

        logger.debug(f"Opened {signal_type} ({strategy_name}) at ${actual_entry:.2f} | Size: {position_size_units:.4f} | Margin: ${margin_used:.2f}")

    def scale_position(self, current_price, scale_data, timestamp):
        """Scale In with specific size if provided"""
        if self.position is None: return

        signal_type = scale_data.get('signal', self.position['type']) # Default to current type
        if self.position['type'] != signal_type: return

        if signal_type == 'LONG':
            actual_entry = current_price * (1 + self.slippage)
        else:
            actual_entry = current_price * (1 - self.slippage)

        # Scaling Size
        size_override = scale_data.get('size_pct')
        if size_override:
            alloc_pct = size_override
        else:
            # Fallback to base logic
            alloc_pct = self.base_position_size_pct * self.position['confidence_at_entry']

        margin_used = self.balance * alloc_pct

        if (self.position['margin'] + margin_used) > (self.balance * 0.5):
            return

        scale_leverage = 1.0
        notional_value = margin_used * scale_leverage
        position_size_units = notional_value / actual_entry

        commission_cost = margin_used * self.commission
        self.balance -= commission_cost

        old_size = self.position['size']
        old_entry = self.position['entry_price']

        new_size = old_size + position_size_units
        new_avg_entry = ((old_size * old_entry) + (position_size_units * actual_entry)) / new_size

        self.position['entry_price'] = new_avg_entry
        self.position['size'] = new_size
        self.position['margin'] += margin_used
        self.position['scale_ins'] += 1

        logger.debug(f"Scaled {signal_type} at ${actual_entry:.2f}. New Avg: ${new_avg_entry:.2f}. Total Size: {new_size:.4f}")

    def close_position(self, exit_price, reason, timestamp):
        if self.position is None: return

        if self.position['type'] == 'LONG':
            actual_exit = exit_price * (1 - self.slippage)
            pnl_usd = (actual_exit - self.position['entry_price']) * self.position['size']
        else:
            actual_exit = exit_price * (1 + self.slippage)
            pnl_usd = (self.position['entry_price'] - actual_exit) * self.position['size']

        commission_cost = self.position['margin'] * self.commission
        pnl_usd -= commission_cost

        self.balance += pnl_usd
        if self.balance < 0: self.balance = 0

        pnl_pct = (pnl_usd / self.position['margin']) * 100 if self.position['margin'] > 0 else 0

        self.trades.append({
            'entry_price': self.position['entry_price'],
            'exit_price': actual_exit,
            'type': self.position['type'],
            'strategy': self.position['strategy'],
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'balance': self.balance,
            'exit_reason': reason,
            'timestamp': timestamp,
            'scale_ins': self.position['scale_ins']
        })

        logger.debug(f"Closed {self.position['strategy']} ({reason}): P&L ${pnl_usd:.2f} ({pnl_pct:.2f}%)")
        self.position = None

    def check_profit_retracement(self, current_price, timestamp):
        if not self.position or self.position['profit_retracement_pct'] is None:
            return False

        if self.position['type'] == 'LONG':
            pnl_val = (current_price - self.position['entry_price']) * self.position['size']
        else:
            pnl_val = (self.position['entry_price'] - current_price) * self.position['size']

        if pnl_val <= 0: return False

        if pnl_val > self.position['highest_unrealized_pnl']:
            self.position['highest_unrealized_pnl'] = pnl_val

        threshold = self.position['highest_unrealized_pnl'] * (1 - self.position['profit_retracement_pct'])

        if pnl_val < threshold:
            self.close_position(current_price, 'Profit Retracement', timestamp)
            return True
        return False

    def check_trailing_stop(self, row):
        if not self.position or not self.position.get('use_trailing'):
            return False

        current_price = row['close']
        timestamp = row['timestamp']

        if self.position['type'] == 'LONG':
            if current_price > self.position['highest_price']:
                self.position['highest_price'] = current_price
        else:
            if current_price < self.position['lowest_price']:
                self.position['lowest_price'] = current_price

        stop_price = 0.0
        val = self.position['trailing_val']

        if self.position['trailing_type'] == 'atr':
            atr = row['atr']
            if self.position['type'] == 'LONG':
                stop_price = self.position['highest_price'] - (atr * val)
                if current_price < stop_price:
                    self.close_position(stop_price, 'Chandelier Exit', timestamp)
                    return True
            else:
                stop_price = self.position['lowest_price'] + (atr * val)
                if current_price > stop_price:
                    self.close_position(stop_price, 'Chandelier Exit', timestamp)
                    return True
        else:
            if self.position['type'] == 'LONG':
                stop_price = self.position['highest_price'] * (1 - val/100)
                if current_price < stop_price:
                    self.close_position(stop_price, 'Trailing Percent', timestamp)
                    return True
            else:
                stop_price = self.position['lowest_price'] * (1 + val/100)
                if current_price > stop_price:
                    self.close_position(stop_price, 'Trailing Percent', timestamp)
                    return True
        return False

    def check_exits(self, row):
        if not self.position: return

        current_price = row['close']
        high = row['high']
        low = row['low']
        timestamp = row['timestamp']

        # 1. Stop Loss
        if self.position['type'] == 'LONG':
            if low <= self.position['stop_loss']:
                self.close_position(self.position['stop_loss'], 'Stop Loss', timestamp)
                return
        else:
            if high >= self.position['stop_loss']:
                self.close_position(self.position['stop_loss'], 'Stop Loss', timestamp)
                return

        # 2. Take Profit
        if self.position['take_profit']:
            if self.position['type'] == 'LONG' and high >= self.position['take_profit']:
                 self.close_position(self.position['take_profit'], 'Take Profit', timestamp)
                 return
            elif self.position['type'] == 'SHORT' and low <= self.position['take_profit']:
                 self.close_position(self.position['take_profit'], 'Take Profit', timestamp)
                 return

        # 3. Trailing
        if self.check_trailing_stop(row):
            return

        # 4. Profit Retracement
        if self.check_profit_retracement(current_price, timestamp):
            return

    def run(self, df, strategy):
        for idx in range(len(df)):
            row = df.iloc[idx]

            if hasattr(strategy, 'prepare_step'):
                strategy.prepare_step(df, idx)

            self.check_exits(row)

            # Check Custom Exit (Dynamic Strategy Exit)
            if self.position and hasattr(strategy, 'check_custom_exit'):
                exit_signal = strategy.check_custom_exit(df, idx, self.position)
                if exit_signal:
                    self.close_position(row['close'], exit_signal, row['timestamp'])
                    continue # Trade closed, skip to next iteration

            if self.position is None and self.balance > 0:
                signal_data = strategy.generate_signal(df, idx)
                if signal_data:
                    stops = strategy.calculate_stops(df, idx, signal_data)
                    self.open_position(
                        signal_data, row['close'], stops['stop_loss'],
                        stops.get('take_profit'),
                        stops.get('use_trailing', False),
                        stops.get('trailing_type', 'percent'),
                        stops.get('trailing_val', 0.0),
                        row['timestamp']
                    )
            elif self.position and self.balance > 0:
                if hasattr(strategy, 'check_scale_in'):
                    scale_data = strategy.check_scale_in(df, idx, self.position)
                    if scale_data:
                        self.scale_position(row['close'], scale_data, row['timestamp'])

            current_equity = self.balance
            if self.position:
                current_price = row['close']
                if self.position['type'] == 'LONG':
                    unrealized_pnl = (current_price - self.position['entry_price']) * self.position['size']
                else:
                    unrealized_pnl = (self.position['entry_price'] - current_price) * self.position['size']
                current_equity += unrealized_pnl

            self.equity_curve.append(current_equity)

        if self.position:
            self.close_position(df.iloc[-1]['close'], 'End of Backtest', df.iloc[-1]['timestamp'])

        return self.calculate_stats()

    def calculate_stats(self):
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
