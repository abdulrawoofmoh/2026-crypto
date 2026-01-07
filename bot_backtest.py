"""
Binance SOL/USDT Futures Trading Bot with Backtesting & Optimization
Features: Trailing Stop, Historical Data Download, Backtesting, Parameter Optimization
Account: $50 USDT | Leverage: 20x | Position Size: 50% of available funds
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import itertools
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SolanaFuturesBot:
    def __init__(self, api_key, api_secret, testnet=True):
        """Initialize the trading bot"""
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': testnet
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

        self.symbol = 'SOL/USDT'
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.highest_price = 0  # For trailing stop in LONG
        self.lowest_price = 0   # For trailing stop in SHORT

        # User-defined parameters
        self.leverage = 20
        self.position_percentage = 0.5

        # Strategy parameters (can be optimized)
        self.params = {
            'timeframe': '15m',
            'lookback': 100,
            'rsi_period': 14,
            'ma_fast': 20,
            'ma_slow': 50,
            'volume_threshold': 1.5,
            'tp_ratio': 2.5,
            'sl_atr_mult': 1.5,
            'trend_strength_min': 0.8,
            'trailing_stop_activation': 1.5,  # Activate trailing after 1.5% profit
            'trailing_stop_distance': 0.8,    # Trail 0.8% behind peak
        }

        self.trade_count = 0
        self.wins = 0
        self.losses = 0

    def download_historical_data(self, years=3):
        """
        Download historical OHLCV data from Binance

        Args:
            years: Number of years of historical data to download

        Returns:
            DataFrame with historical data
        """
        print(f"\n{'='*60}")
        print(f"📥 DOWNLOADING {years} YEARS OF SOL/USDT DATA")
        print(f"{'='*60}\n")

        # Calculate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365 * years)

        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)

        all_candles = []
        current_since = since

        # Binance limit is 1500 candles per request for 15m timeframe
        limit = 1500

        # Calculate total expected candles
        timeframe_minutes = 15
        total_minutes = years * 365 * 24 * 60
        expected_candles = total_minutes // timeframe_minutes

        print(f"Expected candles: ~{expected_candles:,}")
        print(f"Downloading in batches...\n")

        batch_count = 0
        while current_since < int(end_time.timestamp() * 1000):
            try:
                batch_count += 1
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe='15m',
                    since=current_since,
                    limit=limit
                )

                if not ohlcv:
                    break

                all_candles.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1  # Next timestamp

                print(f"Batch {batch_count}: Downloaded {len(ohlcv)} candles | Total: {len(all_candles):,}", end='\r')

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"\nError downloading data: {e}")
                break

        # Create DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        print(f"\n\n✓ Download Complete!")
        print(f"Total Candles: {len(df):,}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"{'='*60}\n")

        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Moving Averages
        df['ma_fast'] = df['close'].rolling(window=self.params['ma_fast']).mean()
        df['ma_slow'] = df['close'].rolling(window=self.params['ma_slow']).mean()

        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()

        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Trend strength
        df['trend_strength'] = abs(df['ma_fast'] - df['ma_slow']) / df['close'] * 100

        # Momentum
        df['momentum'] = df['close'].pct_change(5) * 100

        return df

    def generate_signal(self, df, idx):
        """Generate trading signal at specific index"""
        if idx < self.params['lookback']:
            return None

        latest = df.iloc[idx]
        prev = df.iloc[idx - 1]
        prev2 = df.iloc[idx - 2]

        # Volume confirmation
        if latest['volume_ratio'] < self.params['volume_threshold']:
            return None

        # Trend confirmation
        bullish_trend = (latest['close'] > latest['ma_fast'] > latest['ma_slow'])
        bearish_trend = (latest['close'] < latest['ma_fast'] < latest['ma_slow'])

        # Trend strength
        if latest['trend_strength'] < self.params['trend_strength_min']:
            return None

        # LONG setup
        if bullish_trend:
            if 30 < latest['rsi'] < 50:
                if latest['rsi'] > prev['rsi'] and prev['rsi'] <= prev2['rsi']:
                    if latest['low'] > prev2['low']:
                        if latest['momentum'] > -2:
                            return 'LONG'

        # SHORT setup
        if bearish_trend:
            if 50 < latest['rsi'] < 70:
                if latest['rsi'] < prev['rsi'] and prev['rsi'] >= prev2['rsi']:
                    if latest['high'] < prev2['high']:
                        if latest['momentum'] < 2:
                            return 'SHORT'

        return None

    def check_trailing_stop(self, current_price, position_type):
        """
        Check if trailing stop should be activated and if it's hit

        Returns:
            True if trailing stop is hit, False otherwise
        """
        if position_type == 'LONG':
            # Update highest price
            if current_price > self.highest_price:
                self.highest_price = current_price

            # Calculate profit percentage
            profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100

            # Activate trailing stop after reaching activation threshold
            if profit_pct >= self.params['trailing_stop_activation']:
                # Calculate trailing stop price
                trailing_stop = self.highest_price * (1 - self.params['trailing_stop_distance'] / 100)

                # Check if price hit trailing stop
                if current_price <= trailing_stop:
                    return True

        elif position_type == 'SHORT':
            # Update lowest price
            if current_price < self.lowest_price or self.lowest_price == 0:
                self.lowest_price = current_price

            # Calculate profit percentage
            profit_pct = ((self.entry_price - current_price) / self.entry_price) * 100

            # Activate trailing stop after reaching activation threshold
            if profit_pct >= self.params['trailing_stop_activation']:
                # Calculate trailing stop price
                trailing_stop = self.lowest_price * (1 + self.params['trailing_stop_distance'] / 100)

                # Check if price hit trailing stop
                if current_price >= trailing_stop:
                    return True

        return False

    def backtest(self, df, initial_balance=50, verbose=False):
        """
        Run backtest on historical data

        Args:
            df: DataFrame with OHLCV data and indicators
            initial_balance: Starting balance in USDT
            verbose: Print detailed trade information

        Returns:
            Dictionary with backtest results
        """
        balance = initial_balance
        position = None
        entry_price = 0
        entry_idx = 0
        stop_loss = 0
        take_profit = 0
        position_size = 0
        highest_price = 0
        lowest_price = 0

        trades = []
        equity_curve = [initial_balance]

        for idx in range(len(df)):
            row = df.iloc[idx]
            current_price = row['close']

            # Manage open position
            if position:
                # Calculate current P&L
                if position == 'LONG':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    leveraged_pnl_pct = pnl_pct * self.leverage

                    # Check trailing stop
                    trailing_hit = self.check_trailing_stop(current_price, 'LONG')

                    # Check stop loss
                    if current_price <= stop_loss:
                        # Stop loss hit
                        pnl = (position_size * self.leverage * 0.5) * (stop_loss - entry_price) / entry_price
                        balance += pnl

                        trades.append({
                            'entry_time': df.iloc[entry_idx]['timestamp'],
                            'exit_time': row['timestamp'],
                            'type': position,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'pnl_pct': leveraged_pnl_pct,
                            'pnl_usd': pnl,
                            'balance': balance,
                            'exit_reason': 'Stop Loss'
                        })

                        position = None

                    # Check take profit
                    elif current_price >= take_profit:
                        pnl = (position_size * self.leverage * 0.5) * (take_profit - entry_price) / entry_price
                        balance += pnl

                        trades.append({
                            'entry_time': df.iloc[entry_idx]['timestamp'],
                            'exit_time': row['timestamp'],
                            'type': position,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'pnl_pct': leveraged_pnl_pct,
                            'pnl_usd': pnl,
                            'balance': balance,
                            'exit_reason': 'Take Profit'
                        })

                        position = None

                    # Check trailing stop
                    elif trailing_hit:
                        exit_price = highest_price * (1 - self.params['trailing_stop_distance'] / 100)
                        pnl = (position_size * self.leverage * 0.5) * (exit_price - entry_price) / entry_price
                        balance += pnl

                        trades.append({
                            'entry_time': df.iloc[entry_idx]['timestamp'],
                            'exit_time': row['timestamp'],
                            'type': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pct': ((exit_price - entry_price) / entry_price * 100) * self.leverage,
                            'pnl_usd': pnl,
                            'balance': balance,
                            'exit_reason': 'Trailing Stop'
                        })

                        position = None

                elif position == 'SHORT':
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    leveraged_pnl_pct = pnl_pct * self.leverage

                    # Check trailing stop
                    trailing_hit = self.check_trailing_stop(current_price, 'SHORT')

                    # Check stop loss
                    if current_price >= stop_loss:
                        pnl = (position_size * self.leverage * 0.5) * (entry_price - stop_loss) / entry_price
                        balance += pnl

                        trades.append({
                            'entry_time': df.iloc[entry_idx]['timestamp'],
                            'exit_time': row['timestamp'],
                            'type': position,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'pnl_pct': leveraged_pnl_pct,
                            'pnl_usd': pnl,
                            'balance': balance,
                            'exit_reason': 'Stop Loss'
                        })

                        position = None

                    # Check take profit
                    elif current_price <= take_profit:
                        pnl = (position_size * self.leverage * 0.5) * (entry_price - take_profit) / entry_price
                        balance += pnl

                        trades.append({
                            'entry_time': df.iloc[entry_idx]['timestamp'],
                            'exit_time': row['timestamp'],
                            'type': position,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'pnl_pct': leveraged_pnl_pct,
                            'pnl_usd': pnl,
                            'balance': balance,
                            'exit_reason': 'Take Profit'
                        })

                        position = None

                    # Check trailing stop
                    elif trailing_hit:
                        exit_price = lowest_price * (1 + self.params['trailing_stop_distance'] / 100)
                        pnl = (position_size * self.leverage * 0.5) * (entry_price - exit_price) / entry_price
                        balance += pnl

                        trades.append({
                            'entry_time': df.iloc[entry_idx]['timestamp'],
                            'exit_time': row['timestamp'],
                            'type': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pct': ((entry_price - exit_price) / entry_price * 100) * self.leverage,
                            'pnl_usd': pnl,
                            'balance': balance,
                            'exit_reason': 'Trailing Stop'
                        })

                        position = None

            # Look for new entry signal
            if not position and balance > 0:
                signal = self.generate_signal(df, idx)

                if signal:
                    position = signal
                    entry_price = current_price
                    entry_idx = idx
                    position_size = current_price  # Simplified for backtest

                    # Set stops
                    atr = row['atr']
                    if position == 'LONG':
                        stop_loss = entry_price - (atr * self.params['sl_atr_mult'])
                        take_profit = entry_price + (atr * self.params['sl_atr_mult'] * self.params['tp_ratio'])
                        highest_price = entry_price
                        lowest_price = 0
                    else:
                        stop_loss = entry_price + (atr * self.params['sl_atr_mult'])
                        take_profit = entry_price - (atr * self.params['sl_atr_mult'] * self.params['tp_ratio'])
                        lowest_price = entry_price
                        highest_price = 0

            equity_curve.append(balance)

        # Calculate statistics
        if not trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'final_balance': initial_balance,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'trades': []
            }

        trades_df = pd.DataFrame(trades)
        wins = len(trades_df[trades_df['pnl_usd'] > 0])
        losses = len(trades_df[trades_df['pnl_usd'] <= 0])
        win_rate = (wins / len(trades)) * 100 if trades else 0

        total_return = ((balance - initial_balance) / initial_balance) * 100

        # Max drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified)
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = (returns.mean() / returns.std()) if returns.std() > 0 else 0

        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'final_balance': balance,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0,
            'avg_loss': trades_df[trades_df['pnl_usd'] <= 0]['pnl_usd'].mean() if losses > 0 else 0,
            'trades': trades
        }

    def optimize_parameters(self, df, initial_balance=50):
        """
        Optimize strategy parameters using grid search

        Args:
            df: DataFrame with historical data
            initial_balance: Starting balance

        Returns:
            Best parameters and results
        """
        print(f"\n{'='*60}")
        print(f"🔧 PARAMETER OPTIMIZATION")
        print(f"{'='*60}\n")

        # Parameter ranges to test
        param_grid = {
            'ma_fast': [15, 20, 25],
            'ma_slow': [45, 50, 55],
            'rsi_period': [14, 16],
            'volume_threshold': [1.5],
            'tp_ratio': [2.0, 2.5],
            'sl_atr_mult': [1.2, 1.5],
            'trailing_stop_activation': [1.5],
            'trailing_stop_distance': [0.8],
        }

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        print(f"Testing {len(combinations)} parameter combinations...")
        print(f"This may take a few minutes...\n")

        best_result = None
        best_params = None
        best_score = -float('inf')

        results = []

        for i, combo in enumerate(combinations):
            # Update parameters
            params_dict = dict(zip(keys, combo))
            for key, value in params_dict.items():
                self.params[key] = value

            # Recalculate indicators with new parameters
            df_test = self.calculate_indicators(df.copy())

            # Run backtest
            result = self.backtest(df_test, initial_balance, verbose=False)

            # Score based on: win rate (40%), total return (30%), sharpe (20%), max DD (10%)
            score = (result['win_rate'] * 0.4 +
                    result['total_return'] * 0.3 +
                    result['sharpe_ratio'] * 10 * 0.2 -
                    abs(result['max_drawdown']) * 0.1)

            result['params'] = params_dict
            result['score'] = score
            results.append(result)

            if score > best_score and result['total_trades'] >= 20:
                best_score = score
                best_result = result
                best_params = params_dict

            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Tested {i+1}/{len(combinations)} combinations...", end='\r')

        print(f"\n\n✓ Optimization Complete!\n")

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Display top 5 results
        print("Top 5 Parameter Combinations:")
        print("-" * 60)
        for i, res in enumerate(results[:5], 1):
            print(f"\n#{i} - Score: {res['score']:.2f}")
            print(f"Win Rate: {res['win_rate']:.1f}% | Return: {res['total_return']:.1f}% | "
                  f"Trades: {res['total_trades']}")
            print(f"Params: {res['params']}")

        return best_params, best_result, results

    def print_backtest_results(self, results):
        """Print formatted backtest results"""
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTS")
        print(f"{'='*60}\n")

        print(f"Total Trades:      {results['total_trades']}")
        print(f"Wins:              {results['wins']}")
        print(f"Losses:            {results['losses']}")
        print(f"Win Rate:          {results['win_rate']:.2f}%")
        print(f"\nFinal Balance:     ${results['final_balance']:.2f}")
        print(f"Total Return:      {results['total_return']:.2f}%")
        print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}")

        if results['wins'] > 0:
            print(f"\nAverage Win:       ${results['avg_win']:.2f}")
        if results['losses'] > 0:
            print(f"Average Loss:      ${results['avg_loss']:.2f}")

        print(f"\n{'='*60}")

        # Show last 10 trades
        if results['trades']:
            print(f"\nLast 10 Trades:")
            print("-" * 60)
            trades_df = pd.DataFrame(results['trades'])
            last_10 = trades_df.tail(10)

            for _, trade in last_10.iterrows():
                result_symbol = "✓" if trade['pnl_usd'] > 0 else "✗"
                print(f"{result_symbol} {trade['type']:5s} | "
                      f"Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f} | "
                      f"P&L: {trade['pnl_pct']:+.1f}% (${trade['pnl_usd']:+.2f}) | "
                      f"{trade['exit_reason']}")

        print(f"\n{'='*60}\n")


# Main execution functions
def run_backtest_and_optimization():
    """Run complete backtest and optimization workflow"""
    print("\n" + "="*60)
    print("BACKTEST & OPTIMIZATION SYSTEM")
    print("="*60)

    # Initialize bot (API keys not needed for backtest)
    bot = SolanaFuturesBot(
        api_key="not_needed_for_backtest",
        api_secret="not_needed_for_backtest",
        testnet=True
    )

    # Step 1: Download historical data
    print("\n[1/4] Downloading historical data...")
    df = bot.download_historical_data(years=3)

    if df.empty:
        print("\n⚠️ No data downloaded. Check internet connection or API restrictions.")
        print("Cannot proceed with backtest.")
        return bot, df, {}, {}

    # Step 2: Calculate indicators
    print("[2/4] Calculating technical indicators...")
    df = bot.calculate_indicators(df)

    # Step 3: Run backtest with current parameters
    print("[3/4] Running backtest with default parameters...")
    results = bot.backtest(df, initial_balance=50)
    bot.print_backtest_results(results)

    # Step 4: Optimize parameters
    print("[4/4] Starting parameter optimization...")
    best_params, best_results, all_results = bot.optimize_parameters(df, initial_balance=50)

    print(f"\n{'='*60}")
    print(f"🏆 BEST PARAMETERS FOUND")
    print(f"{'='*60}\n")
    print(f"Parameters: {best_params}")
    print(f"\nPerformance:")
    bot.print_backtest_results(best_results)

    return bot, df, best_params, best_results


if __name__ == "__main__":
    # Run backtest and optimization
    bot, df, best_params, best_results = run_backtest_and_optimization()

    print("\n" + "="*60)
    print("✓ Analysis Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review the optimized parameters above")
    print("2. Update your bot configuration with best parameters")
    print("3. Run forward test on recent data")
    print("4. If satisfied, deploy for live trading")
    print("="*60 + "\n")
