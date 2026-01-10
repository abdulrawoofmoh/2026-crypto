"""Data loader - Download historical OHLCV data from Binance"""

import ccxt
import pandas as pd
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

class DataLoader:
    """Download and manage historical market data"""

    def __init__(self, exchange_id='binanceus', testnet=False):
        # Default to binanceus for US-based IP addresses (common in cloud envs)
        # We use public data, so we don't strictly need API keys.
        self.exchange_id = exchange_id
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                # 'options': {'defaultType': 'future'} # binanceus might not support future in ccxt cleanly, sticking to spot for price action
            })
        except AttributeError:
             logger.warning(f"Exchange {exchange_id} not found, falling back to binanceus")
             self.exchange = ccxt.binanceus({'enableRateLimit': True})

        if testnet:
            if 'binance' in exchange_id:
                self.exchange.set_sandbox_mode(True)

    def download_data(self, symbol, timeframe, start_date, end_date):
        """
        Download OHLCV data
        """
        # Adjust symbol for Binance US if needed (SOL/USDT is standard)
        logger.info(f"Downloading {symbol} {timeframe} from {start_date} to {end_date} using {self.exchange.id}")

        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        all_candles = []
        current_since = since

        # Parse timeframe to milliseconds
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000

        while current_since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                if not ohlcv:
                    logger.warning(f"No data returned from {current_since}")
                    break

                # Check for overlaps
                if len(all_candles) > 0 and ohlcv[0][0] <= all_candles[-1][0]:
                     # Move strictly past the last candle we have
                     current_since = all_candles[-1][0] + 1
                     continue

                all_candles.extend(ohlcv)

                last_candle_ts = ohlcv[-1][0]
                current_since = last_candle_ts + 1

                logger.info(f"Downloaded {len(ohlcv)} candles. Total: {len(all_candles)}. Last date: {datetime.fromtimestamp(last_candle_ts/1000)}")

                if len(ohlcv) < 1:
                     break

            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                time.sleep(2)
                break

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        # Filter strictly by start/end date
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

        logger.info(f"Final dataset: {len(df)} candles")
        return df
