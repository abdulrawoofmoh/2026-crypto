"""
Jules Trading System - Main Entry Point
Run backtest and analyze SOL/USDT futures strategy
"""

import sys
import os
import logging
from datetime import datetime

# Add bot directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import *
from bot.data.binance_client import BinanceClient
from bot.data.data_fetcher import DataFetcher
from bot.indicators.technical import calculate_all_indicators
from bot.detectors.liquidity import (calculate_volume_profile, detect_swing_points,
                                      find_swing_clusters, detect_fair_value_gaps)
from bot.detectors.smc import apply_all_smc_detectors
from bot.strategy import TradingStrategy
from backtest.engine import BacktestEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/jules.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_results(results: dict):
    """Print backtest results in formatted way"""
    print(f"\n{'='*60}")
    print(f"📊 JULES BACKTEST RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Total Trades:      {results['total_trades']}")
    print(f"Wins:              {results.get('wins', 0)}")
    print(f"Losses:            {results.get('losses', 0)}")
    print(f"Win Rate:          {results.get('win_rate', 0):.2f}%")
    print(f"\nInitial Balance:   ${BACKTEST_CONFIG['starting_balance']:.2f}")
    print(f"Final Balance:     ${results['final_balance']:.2f}")
    print(f"Total Return:      {results.get('total_return', 0):.2f}%")
    print(f"Max Drawdown:      {results.get('max_drawdown', 0):.2f}%")
    print(f"Sharpe Ratio:      {results.get('sharpe_ratio', 0):.2f}")
    
    if results.get('wins', 0) > 0:
        print(f"\nAverage Win:       ${results.get('avg_win', 0):.2f}")
    if results.get('losses', 0) > 0:
        print(f"Average Loss:      ${results.get('avg_loss', 0):.2f}")
    
    print(f"\n{'='*60}")
    
    # Show last 10 trades
    if results.get('trades', []):
        print(f"\n📈 Last 10 Trades:")
        print("-" * 60)
        
        import pandas as pd
        trades_df = pd.DataFrame(results['trades'])
        last_10 = trades_df.tail(10)
        
        for _, trade in last_10.iterrows():
            result_symbol = "✓" if trade['pnl_usd'] > 0 else "✗"
            print(f"{result_symbol} {trade['type']:5s} | "
                  f"Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f} | "
                  f"P&L: {trade['pnl_pct']:+.1f}% (${trade['pnl_usd']:+.2f}) | "
                  f"{trade['exit_reason']}")
        
        print("-" * 60)
    
    print(f"\n{'='*60}\n")


def run_full_backtest():
    """Run complete backtest with Jules system"""
    
    print("\n" + "="*60)
    print("🚀 JULES TRADING SYSTEM - BACKTEST MODE")
    print("="*60)
    print(f"\nSymbol:           {TRADING_CONFIG['symbol']}")
    print(f"Timeframe:        {TRADING_CONFIG['timeframe_primary']}")
    print(f"Initial Balance:  ${BACKTEST_CONFIG['starting_balance']}")
    print(f"Leverage:         {TRADING_CONFIG['leverage']}x")
    print(f"Position Size:    {TRADING_CONFIG['position_size']*100}%")
    print(f"Historical Data:  {DATA_CONFIG['historical_years']} years")
    print("="*60 + "\n")
    
    # Initialize client (no API keys needed for backtest)
    client = BinanceClient(api_key='', api_secret='', testnet=True)
    fetcher = DataFetcher(client)
    
    # Download historical data
    print("[1/7] Downloading historical data...")
    symbol = TRADING_CONFIG['symbol']
    
    df_15m = fetcher.download_historical_data(
        symbol, 
        TRADING_CONFIG['timeframe_primary'], 
        DATA_CONFIG['historical_years']
    )
    
    df_1h = fetcher.download_historical_data(
        symbol, 
        TRADING_CONFIG['timeframe_trend'], 
        DATA_CONFIG['historical_years']
    )
    
    df_4h = fetcher.download_historical_data(
        symbol, 
        TRADING_CONFIG['timeframe_major'], 
        DATA_CONFIG['historical_years']
    )
    
    print(f"✓ Downloaded {len(df_15m):,} candles (15m)")
    print(f"✓ Downloaded {len(df_1h):,} candles (1h)")
    print(f"✓ Downloaded {len(df_4h):,} candles (4h)")
    
    # Calculate technical indicators
    print("\n[2/7] Calculating technical indicators...")
    df_15m = calculate_all_indicators(df_15m, STRATEGY_PARAMS)
    df_1h = calculate_all_indicators(df_1h, STRATEGY_PARAMS)
    df_4h = calculate_all_indicators(df_4h, STRATEGY_PARAMS)
    print("✓ Technical indicators calculated")
    
    # Detect liquidity levels
    print("\n[3/7] Detecting liquidity levels...")
    df_15m = detect_swing_points(df_15m)
    df_15m = detect_fair_value_gaps(df_15m)
    
    # Calculate volume profile for entire dataset
    volume_profile = calculate_volume_profile(df_15m, bins=STRATEGY_PARAMS['volume_profile_bins'])
    print(f"✓ Volume Profile: POC=${volume_profile.get('poc', 0):.2f}")
    
    # Find swing clusters
    swing_clusters = find_swing_clusters(df_15m, STRATEGY_PARAMS['swing_cluster_distance'])
    print(f"✓ Found {len(swing_clusters)} swing clusters")
    
    # Apply SMC detectors
    print("\n[4/7] Applying Smart Money Concepts...")
    df_15m = apply_all_smc_detectors(df_15m, STRATEGY_PARAMS)
    print("✓ SMC detection complete")
    
    # Initialize strategy
    print("\n[5/7] Initializing trading strategy...")
    strategy = TradingStrategy(
        params=STRATEGY_PARAMS,
        scoring_weights=SCORING_WEIGHTS,
        thresholds=PROBABILITY_THRESHOLDS,
        penalties=SCORING_PENALTIES
    )
    print("✓ Strategy initialized")
    
    # Initialize backtest engine
    print("\n[6/7] Running backtest...")
    engine = BacktestEngine(
        initial_balance=BACKTEST_CONFIG['starting_balance'],
        leverage=TRADING_CONFIG['leverage'],
        position_size=TRADING_CONFIG['position_size'],
        commission=BACKTEST_CONFIG['commission'],
        slippage=BACKTEST_CONFIG['slippage']
    )
    
    # Run backtest
    results = engine.run_backtest(
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
        strategy=strategy,
        volume_profiles={'15m': volume_profile},
        trailing_activation=STRATEGY_PARAMS['trailing_stop_activation'],
        trailing_distance=STRATEGY_PARAMS['trailing_stop_distance']
    )
    
    print("✓ Backtest complete!")
    
    # Print results
    print("\n[7/7] Analyzing results...")
    print_results(results)
    
    # Save results
    import pandas as pd
    if results.get('trades', []):
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('logs/backtest_trades.csv', index=False)
        print("💾 Trade log saved to: logs/backtest_trades.csv")
    
    if results.get('equity_curve', []):
        equity_df = pd.DataFrame({'equity': results['equity_curve']})
        equity_df.to_csv('logs/equity_curve.csv', index=False)
        print("💾 Equity curve saved to: logs/equity_curve.csv")
    
    # Compare to old results
    print("\n" + "="*60)
    print("📊 COMPARISON TO OLD STRATEGY")
    print("="*60)
    print(f"Old Strategy:")
    print(f"  Win Rate:  20.0%")
    print(f"  Return:    -102.6%")
    print(f"  Trades:    15 (over 3 years)")
    print(f"\nJules System:")
    print(f"  Win Rate:  {results['win_rate']:.1f}%")
    print(f"  Return:    {results['total_return']:.1f}%")
    print(f"  Trades:    {results['total_trades']} (over 3 years)")
    
    improvement = results['win_rate'] - 20.0
    print(f"\n🎯 Win Rate Improvement: {improvement:+.1f} percentage points")
    
    if results['win_rate'] >= 70:
        print("✅ TARGET ACHIEVED: 70%+ win rate!")
    else:
        print(f"⚠️  Target: 70% | Current: {results['win_rate']:.1f}% | Gap: {70-results['win_rate']:.1f}%")
    
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Run backtest
        results = run_full_backtest()
        
        print("\n✅ All done! Check the logs folder for detailed results.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user")
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Check logs/jules.log for details")