"""
Data Fetcher
Downloads and manages historical and real-time market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging
from typing import Dict, List, Optional
from .binance_client import BinanceClient

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles downloading and caching market data"""
    
    def __init__(self, client: BinanceClient, cache_dir: str = 'data/cache/'):
        """
        Initialize data fetcher
        
        Args:
            client: BinanceClient instance
            cache_dir: Directory to cache downloaded data
        """
        self.client = client
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_historical_data(self, symbol: str, timeframe: str, years: int = 3) -> pd.DataFrame:
        """
        Download historical OHLCV data
        
        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            timeframe: Timeframe (e.g., '15m', '1h', '4h')
            years: Number of years of data to download
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading {years} years of {symbol} {timeframe} data...")
        
        # Check cache first
        cache_file = f"{self.cache_dir}{symbol.replace('/', '_')}_{timeframe}_{years}y.csv"
        if os.path.exists(cache_file):
            logger.info(f"Loading from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365 * years)
        since = int(start_time.timestamp() * 1000)
        
        all_candles = []
        current_since = since
        batch_count = 0
        
        # Download in batches
        while current_since < int(end_time.timestamp() * 1000):
            batch_count += 1
            
            try:
                ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=1500, since=current_since)
                
                if not ohlcv:
                    break
                
                all_candles.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                if batch_count % 10 == 0:
                    logger.info(f"Downloaded {len(all_candles):,} candles...")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error downloading batch: {e}")
                break
        
        # Create DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        logger.info(f"Downloaded {len(df):,} candles and saved to cache")
        
        return df
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple timeframes
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes (e.g., ['15m', '1h', '4h'])
            years: Number of years of data
            
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        data = {}
        
        for tf in timeframes:
            logger.info(f"Fetching {tf} data...")
            df = self.download_historical_data(symbol, tf, years)
            data[tf] = df
        
        return data
    
    def get_latest_candles(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Get latest candles (for live trading)
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Number of recent candles
            
        Returns:
            DataFrame with recent OHLCV data
        """
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            logger.error(f"Error fetching latest candles: {e}")
            return pd.DataFrame()
    
    def get_order_book_snapshot(self, symbol: str, levels: int = 20) -> Dict:
        """
        Get current order book snapshot
        
        Args:
            symbol: Trading pair
            levels: Number of price levels
            
        Returns:
            Dictionary with bids, asks, and analysis
        """
        try:
            order_book = self.client.fetch_order_book(symbol, limit=levels)
            
            # Calculate bid/ask volumes
            bid_volume = sum([bid[1] for bid in order_book['bids']])
            ask_volume = sum([ask[1] for ask in order_book['asks']])
            
            # Calculate imbalance ratio
            imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 0
            
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'imbalance_ratio': imbalance_ratio,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return {}
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get recent trades
        
        Args:
            symbol: Trading pair
            limit: Number of trades
            
        Returns:
            DataFrame with recent trades
        """
        try:
            trades = self.client.fetch_recent_trades(symbol, limit=limit)
            
            if not trades:
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            return pd.DataFrame()
    
    def analyze_large_orders(self, trades_df: pd.DataFrame, threshold_multiplier: float = 5.0) -> List[Dict]:
        """
        Identify large orders from recent trades
        
        Args:
            trades_df: DataFrame with recent trades
            threshold_multiplier: Multiplier for average size
            
        Returns:
            List of large order dictionaries
        """
        if trades_df.empty:
            return []
        
        # Calculate average trade size
        avg_size = trades_df['amount'].mean()
        threshold = avg_size * threshold_multiplier
        
        # Find large orders
        large_orders = trades_df[trades_df['amount'] >= threshold]
        
        results = []
        for _, order in large_orders.iterrows():
            results.append({
                'timestamp': order['timestamp'],
                'price': order['price'],
                'amount': order['amount'],
                'side': order['side'],
                'size_ratio': order['amount'] / avg_size
            })
        
        return results