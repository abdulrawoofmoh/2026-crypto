"""
Backtest Engine
Simulates trading on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for strategy validation"""
    
    def __init__(self, initial_balance: float, leverage: int, position_size: float,
                 commission: float = 0.0004, slippage: float = 0.0005):
        """
        Initialize backtest engine
        
        Args:
            initial_balance: Starting balance in USDT
            leverage: Leverage multiplier
            position_size: Position size as fraction of balance
            commission: Commission per trade
            slippage: Slippage per trade
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.position_size_pct = position_size
        self.commission = commission
        self.slippage = slippage
        
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.highest_price = 0
        self.lowest_price = 0
        
        self.trades = []
        self.equity_curve = [initial_balance]
        
    def open_position(self, signal: Dict, current_price: float):
        """
        Open a new position
        
        Args:
            signal: Signal dictionary from strategy
            current_price: Current market price
        """
        if self.position is not None:
            return
        
        # Apply slippage
        if signal['signal'] == 'LONG':
            entry_price = current_price * (1 + self.slippage)
        else:
            entry_price = current_price * (1 - self.slippage)
        
        # Calculate position size
        margin = self.balance * self.position_size_pct
        position_value = margin * self.leverage
        self.position_size = position_value / entry_price
        
        # Deduct commission
        commission_cost = position_value * self.commission
        self.balance -= commission_cost
        
        # Set position details
        self.position = signal['signal']
        self.entry_price = entry_price
        self.stop_loss = signal['stop_loss']
        self.take_profit = signal['take_profit']
        
        # Initialize trailing stop tracking
        if signal['signal'] == 'LONG':
            self.highest_price = entry_price
            self.lowest_price = 0
        else:
            self.lowest_price = entry_price
            self.highest_price = 0
        
        logger.debug(f"Opened {self.position} at ${entry_price:.2f}, Size: {self.position_size:.3f}")
    
    def check_trailing_stop(self, current_price: float, trailing_activation: float, 
                           trailing_distance: float) -> bool:
        """
        Check if trailing stop is hit
        
        Args:
            current_price: Current price
            trailing_activation: Activation threshold percentage
            trailing_distance: Trail distance percentage
            
        Returns:
            True if trailing stop hit
        """
        if self.position == 'LONG':
            # Update highest price
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Calculate profit
            profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            
            # Activate trailing stop
            if profit_pct >= trailing_activation:
                trailing_stop = self.highest_price * (1 - trailing_distance / 100)
                if current_price <= trailing_stop:
                    return True
        
        elif self.position == 'SHORT':
            # Update lowest price
            if current_price < self.lowest_price or self.lowest_price == 0:
                self.lowest_price = current_price
            
            # Calculate profit
            profit_pct = ((self.entry_price - current_price) / self.entry_price) * 100
            
            # Activate trailing stop
            if profit_pct >= trailing_activation:
                trailing_stop = self.lowest_price * (1 + trailing_distance / 100)
                if current_price >= trailing_stop:
                    return True
        
        return False
    
    def close_position(self, exit_price: float, exit_reason: str, timestamp):
        """
        Close current position
        
        Args:
            exit_price: Exit price
            exit_reason: Reason for exit
            timestamp: Exit timestamp
        """
        if self.position is None:
            return
        
        # Apply slippage
        if self.position == 'LONG':
            actual_exit = exit_price * (1 - self.slippage)
        else:
            actual_exit = exit_price * (1 + self.slippage)
        
        # Calculate P&L
        if self.position == 'LONG':
            pnl_pct = ((actual_exit - self.entry_price) / self.entry_price) * 100
        else:
            pnl_pct = ((self.entry_price - actual_exit) / self.entry_price) * 100
        
        # With leverage
        leveraged_pnl_pct = pnl_pct * self.leverage
        
        # Calculate USD P&L
        position_value = self.position_size * self.entry_price
        pnl_usd = (position_value * leveraged_pnl_pct) / 100
        
        # Deduct commission
        commission_cost = (self.position_size * actual_exit) * self.commission
        pnl_usd -= commission_cost
        
        # Update balance
        self.balance += pnl_usd
        
        # Record trade
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': actual_exit,
            'type': self.position,
            'pnl_pct': leveraged_pnl_pct,
            'pnl_usd': pnl_usd,
            'balance': self.balance,
            'exit_reason': exit_reason,
            'timestamp': timestamp
        })
        
        # Reset position
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        
        logger.debug(f"Closed position: {exit_reason}, P&L: ${pnl_usd:.2f}")
    
    def update_position(self, row, trailing_activation: float, trailing_distance: float):
        """
        Update open position (check stops)
        
        Args:
            row: Current candle data
            trailing_activation: Trailing stop activation threshold
            trailing_distance: Trailing stop distance
        """
        if self.position is None:
            return
        
        current_price = row['close']
        high = row['high']
        low = row['low']
        
        if self.position == 'LONG':
            # Check stop loss
            if low <= self.stop_loss:
                self.close_position(self.stop_loss, 'Stop Loss', row['timestamp'])
                return
            
            # Check take profit
            if high >= self.take_profit:
                self.close_position(self.take_profit, 'Take Profit', row['timestamp'])
                return
            
            # Check trailing stop
            if self.check_trailing_stop(current_price, trailing_activation, trailing_distance):
                exit_price = self.highest_price * (1 - trailing_distance / 100)
                self.close_position(exit_price, 'Trailing Stop', row['timestamp'])
                return
        
        elif self.position == 'SHORT':
            # Check stop loss
            if high >= self.stop_loss:
                self.close_position(self.stop_loss, 'Stop Loss', row['timestamp'])
                return
            
            # Check take profit
            if low <= self.take_profit:
                self.close_position(self.take_profit, 'Take Profit', row['timestamp'])
                return
            
            # Check trailing stop
            if self.check_trailing_stop(current_price, trailing_activation, trailing_distance):
                exit_price = self.lowest_price * (1 + trailing_distance / 100)
                self.close_position(exit_price, 'Trailing Stop', row['timestamp'])
                return
    
    def run_backtest(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                     strategy, volume_profiles: Dict = None, 
                     trailing_activation: float = 1.5, trailing_distance: float = 0.8) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df_15m: 15-minute DataFrame with all indicators
            df_1h: 1-hour DataFrame
            df_4h: 4-hour DataFrame
            strategy: TradingStrategy instance
            volume_profiles: Dictionary of volume profiles
            trailing_activation: Trailing stop activation %
            trailing_distance: Trailing stop distance %
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        for idx in range(len(df_15m)):
            row = df_15m.iloc[idx]
            
            # Update open position
            if self.position:
                self.update_position(row, trailing_activation, trailing_distance)
            
            # Look for new entry signal (only if no position)
            if self.position is None and self.balance > 0:
                # Get volume profile for current period
                volume_profile = None
                if volume_profiles and idx >= 500:
                    # Use volume profile from last 500 candles
                    recent_df = df_15m.iloc[max(0, idx-500):idx]
                    from bot.detectors.liquidity import calculate_volume_profile
                    volume_profile = calculate_volume_profile(recent_df)
                
                # Generate signal
                signal = strategy.generate_signal(df_15m, df_1h, df_4h, idx, 
                                                 volume_profile=volume_profile)
                
                if signal:
                    self.open_position(signal, row['close'])
            
            # Update equity curve
            current_equity = self.balance
            if self.position:
                # Add unrealized P&L
                current_price = row['close']
                if self.position == 'LONG':
                    unrealized_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                else:
                    unrealized_pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                
                position_value = self.position_size * self.entry_price
                unrealized_pnl = (position_value * unrealized_pnl_pct * self.leverage) / 100
                current_equity += unrealized_pnl
            
            self.equity_curve.append(current_equity)
        
        # Close any remaining position
        if self.position:
            last_row = df_15m.iloc[-1]
            self.close_position(last_row['close'], 'End of Backtest', last_row['timestamp'])
        
        # Calculate statistics
        return self.calculate_statistics()
    
    def calculate_statistics(self) -> Dict:
        """Calculate backtest performance statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'final_balance': self.balance,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        wins = len(trades_df[trades_df['pnl_usd'] > 0])
        losses = len(trades_df[trades_df['pnl_usd'] <= 0])
        win_rate = (wins / len(self.trades)) * 100 if self.trades else 0
        
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Max drawdown
        equity_series = pd.Series(self.equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Average win/loss
        avg_win = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl_usd'] <= 0]['pnl_usd'].mean() if losses > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'final_balance': self.balance,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }