"""Performance metrics and reporting"""

import pandas as pd

def print_results(results, strategy_name='Strategy'):
    """Print formatted backtest results"""
    print(f"\n{'='*60}")
    print(f"📊 {strategy_name.upper()} RESULTS")
    print(f"{'='*60}\n")

    print(f"Total Trades:      {results['total_trades']}")
    print(f"Wins:              {results['wins']}")
    print(f"Losses:            {results['losses']}")
    print(f"Win Rate:          {results['win_rate']:.2f}%")
    print(f"\nInitial Balance:   $50.00")
    print(f"Final Balance:     ${results['final_balance']:.2f}")
    print(f"Total Return:      {results['total_return']:.2f}%")
    print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}")

    if results['wins'] > 0:
        print(f"\nAverage Win:       ${results['avg_win']:.2f}")
    if results['losses'] > 0:
        print(f"Average Loss:      ${results['avg_loss']:.2f}")

    print(f"\n{'='*60}")

    if results.get('trades'):
        print(f"\nSample Trades (Last 5):")
        print("-" * 60)
        trades_df = pd.DataFrame(results['trades'])
        for _, trade in trades_df.tail(5).iterrows():
            result_symbol = "✓" if trade['pnl_usd'] > 0 else "✗"
            scale_info = f"(Scaled {trade['scale_ins']}x)" if trade['scale_ins'] > 0 else ""
            print(f"{result_symbol} {trade['type']:5s} | Entry: ${trade['entry_price']:.2f} {scale_info} → "
                  f"Exit: ${trade['exit_price']:.2f} | P&L: {trade['pnl_pct']:+.1f}% "
                  f"(${trade['pnl_usd']:+.2f}) | {trade['exit_reason']}")

    print(f"\n{'='*60}\n")

def save_results(results, filename):
    """Save results to CSV"""
    if results.get('trades'):
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(filename, index=False)
        print(f"💾 Results saved to: {filename}")
