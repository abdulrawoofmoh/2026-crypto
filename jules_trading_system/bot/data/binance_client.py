"""
Binance API Client
Handles all interactions with Binance exchange
"""

import ccxt
import time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BinanceClient:
    """Wrapper for Binance exchange operations"""
    
    def __init__(self, api_key: str = '', api_secret: str = '', testnet: bool = True):
        """
        Initialize Binance client
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True, live trading if False
        """
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
            
        logger.info(f"Binance client initialized (testnet={testnet})")
        
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500, since: Optional[int] = None) -> List:
        """
        Fetch OHLCV candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            timeframe: Timeframe (e.g., '15m', '1h', '4h')
            limit: Number of candles to fetch
            since: Timestamp to start from (milliseconds)
            
        Returns:
            List of OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            logger.debug(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return []
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Fetch order book (bids and asks)
        
        Args:
            symbol: Trading pair
            limit: Number of levels to fetch
            
        Returns:
            Dictionary with 'bids' and 'asks'
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return {'bids': [], 'asks': []}
    
    def fetch_recent_trades(self, symbol: str, limit: int = 100) -> List:
        """
        Fetch recent trades
        
        Args:
            symbol: Trading pair
            limit: Number of trades to fetch
            
        Returns:
            List of recent trades
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            return []
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker information
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dictionary with ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return {}
    
    def fetch_balance(self) -> Dict:
        """
        Fetch account balance
        
        Returns:
            Dictionary with balance information
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    def set_leverage(self, symbol: str, leverage: int):
        """
        Set leverage for futures trading
        
        Args:
            symbol: Trading pair
            leverage: Leverage amount (e.g., 20 for 20x)
        """
        try:
            symbol_for_api = symbol.replace('/', '')
            self.exchange.fapiPrivate_post_leverage({
                'symbol': symbol_for_api,
                'leverage': leverage
            })
            logger.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
    
    def create_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """
        Create market order
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Order information dictionary or None if failed
        """
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"Market order created: {side} {amount} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            return None
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict]:
        """
        Create limit order
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price
            
        Returns:
            Order information dictionary or None if failed
        """
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"Limit order created: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Error creating limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def fetch_open_orders(self, symbol: str) -> List:
        """
        Fetch open orders
        
        Args:
            symbol: Trading pair
            
        Returns:
            List of open orders
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []