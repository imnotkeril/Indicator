"""
Module for collecting data from cryptocurrency exchanges.
Handles connection to exchanges and fetching of OHLCV data.
"""
import os
import sys
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.config import config
from crypto_volatility_indicator.utils.logger import get_data_logger
from crypto_volatility_indicator.utils.helpers import (
    timestamp_to_datetime,
    ensure_directory,
    save_dataframe,
    load_dataframe,
    timeframe_to_seconds
)

# Set up logger
logger = get_data_logger()

class ExchangeDataCollector:
    """
    Collector for exchange market data.

    This class handles connections to cryptocurrency exchanges
    and fetches OHLCV (Open, High, Low, Close, Volume) data.
    """

    def __init__(self, exchange_id=None, api_key=None, api_secret=None, symbols=None, timeframes=None):
        """
        Initialize the exchange data collector.

        Parameters:
        -----------
        exchange_id : str, optional
            Exchange ID (e.g., 'binance', 'coinbase')
            If None, uses the value from config
        api_key : str, optional
            API key for the exchange
            If None, uses the value from config
        api_secret : str, optional
            API secret for the exchange
            If None, uses the value from config
        symbols : list, optional
            List of trading pairs to collect data for (e.g., ['BTC/USDT', 'ETH/USDT'])
            If None, uses the value from config
        timeframes : dict, optional
            Dictionary of timeframes for different volatility levels
            If None, uses the value from config
        """
        # Load configuration if not provided
        self.exchange_id = exchange_id or config.get('exchange.name', 'binance')
        self.api_key = api_key or config.get('exchange.api_key', '')
        self.api_secret = api_secret or config.get('exchange.api_secret', '')
        self.symbols = symbols or config.get('symbols', ['BTC/USDT'])

        self.timeframes = timeframes or {
            'micro': config.get('timeframes.micro', '1m'),
            'meso': config.get('timeframes.meso', '1h'),
            'macro': config.get('timeframes.macro', '1d')
        }

        # Initialize exchange
        self.initialize_exchange()

        # Data storage
        self.data = {}

        # Cache directory
        self.cache_dir = Path('data/cache')
        ensure_directory(self.cache_dir)

        logger.info(f"ExchangeDataCollector initialized for {self.exchange_id} with symbols: {self.symbols}")

    def initialize_exchange(self):
        """Initialize the connection to the exchange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)

            # Create exchange instance
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'timeout': config.get('exchange.timeout', 30000)
            })

            # Set up any exchange-specific settings
            if self.exchange_id == 'binance':
                self.exchange.options['defaultType'] = 'spot'

            logger.info(f"Connected to {self.exchange_id} exchange")

        except Exception as e:
            logger.error(f"Failed to initialize exchange {self.exchange_id}: {e}")
            raise

    def fetch_ohlcv_data(self, symbol, timeframe, since=None, limit=1000):
        """
        Fetch OHLCV data for a specific symbol and timeframe.

        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., 'BTC/USDT')
        timeframe : str
            Timeframe (e.g., '1m', '1h', '1d')
        since : int or datetime, optional
            Start time for data fetching (timestamp or datetime)
        limit : int, optional
            Maximum number of candles to fetch

        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to timestamp if needed
            if isinstance(since, datetime):
                since = int(since.timestamp() * 1000)

            # Fetch data
            logger.info(f"Fetching {timeframe} data for {symbol} from {self.exchange_id}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
            return None

    def fetch_all_data(self, since=None, limit=1000):
        """
        Fetch data for all configured symbols and timeframes.

        Parameters:
        -----------
        since : int or datetime, optional
            Start time for data fetching (timestamp or datetime)
        limit : int, optional
            Maximum number of candles to fetch per request

        Returns:
        --------
        dict
            Dictionary with fetched data
        """
        self.data = {}

        for symbol in self.symbols:
            self.data[symbol] = {}

            for level, timeframe in self.timeframes.items():
                df = self.fetch_ohlcv_data(symbol, timeframe, since, limit)

                if df is not None and not df.empty:
                    self.data[symbol][level] = df
                else:
                    logger.warning(f"No data fetched for {symbol} at {level} level ({timeframe})")

        return self.data

    def fetch_historical_data(self, symbol, timeframe, start_date, end_date=None):
        """
        Fetch historical data for a specific period.

        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., 'BTC/USDT')
        timeframe : str
            Timeframe (e.g., '1m', '1h', '1d')
        start_date : datetime
            Start date for historical data
        end_date : datetime, optional
            End date for historical data (defaults to now)

        Returns:
        --------
        pd.DataFrame
            DataFrame with historical OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()

        # Calculate number of candles to fetch
        timeframe_seconds = timeframe_to_seconds(timeframe)
        total_seconds = (end_date - start_date).total_seconds()
        candles_needed = int(total_seconds / timeframe_seconds)

        # CCXT fetch_ohlcv has a limit, so we may need multiple requests
        max_candles_per_request = 1000
        all_data = []

        current_since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        while current_since < end_timestamp:
            try:
                logger.info(f"Fetching {timeframe} data for {symbol} from {timestamp_to_datetime(current_since, 'ms')}")

                # Fetch data
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=max_candles_per_request)

                if not ohlcv:
                    logger.warning(f"No data returned for {symbol} at {timestamp_to_datetime(current_since, 'ms')}")
                    break

                all_data.extend(ohlcv)

                # Update since for next request
                last_timestamp = ohlcv[-1][0]

                # If we received the same timestamp as before, we're done
                if last_timestamp <= current_since:
                    break

                current_since = last_timestamp

                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                break

        if not all_data:
            logger.warning(f"No historical data fetched for {symbol} ({timeframe})")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Filter to requested date range
        df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

        logger.info(f"Fetched {len(df)} historical {timeframe} candles for {symbol}")
        return df

    def fetch_recent_data(self, symbol, timeframe, lookback_days):
        """
        Fetch recent data for a specific number of days.

        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., 'BTC/USDT')
        timeframe : str
            Timeframe (e.g., '1m', '1h', '1d')
        lookback_days : int
            Number of days to look back

        Returns:
        --------
        pd.DataFrame
            DataFrame with recent OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        return self.fetch_historical_data(symbol, timeframe, start_date, end_date)

    def save_data(self, symbol, level, data=None, format='parquet'):
        """
        Save data to cache.

        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., 'BTC/USDT')
        level : str
            Volatility level ('micro', 'meso', 'macro')
        data : pd.DataFrame, optional
            Data to save (if None, uses self.data[symbol][level])
        format : str, optional
            File format ('csv', 'pickle', 'parquet')

        Returns:
        --------
        Path
            Path to saved file
        """
        # Use provided data or get from instance
        if data is None:
            if symbol not in self.data or level not in self.data[symbol]:
                logger.error(f"No data available for {symbol} at {level} level")
                return None

            data = self.data[symbol][level]

        # Replace '/' in symbol with '_' for filename
        safe_symbol = symbol.replace('/', '_')

        # Create filename
        filename = f"{safe_symbol}_{level}_{datetime.now().strftime('%Y%m%d')}.{format}"
        filepath = self.cache_dir / filename

        # Save data
        return save_dataframe(data, filepath, format)

    def load_cached_data(self, symbol, level, date=None, format='parquet'):
        """
        Load data from cache.

        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., 'BTC/USDT')
        level : str
            Volatility level ('micro', 'meso', 'macro')
        date : datetime or str, optional
            Date to load data for (if None, loads most recent)
        format : str, optional
            File format ('csv', 'pickle', 'parquet')

        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        # Replace '/' in symbol with '_' for filename
        safe_symbol = symbol.replace('/', '_')

        # Find most recent file if date not specified
        if date is None:
            pattern = f"{safe_symbol}_{level}_*.{format}"
            files = list(self.cache_dir.glob(pattern))

            if not files:
                logger.warning(f"No cached data found for {symbol} at {level} level")
                return None

            # Get most recent file
            filepath = max(files, key=lambda p: p.stat().st_mtime)
        else:
            # Format date if it's a datetime
            if isinstance(date, datetime):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = date

            filepath = self.cache_dir / f"{safe_symbol}_{level}_{date_str}.{format}"

            if not filepath.exists():
                logger.warning(f"No cached data found for {symbol} at {level} level for date {date_str}")
                return None

        # Load data
        return load_dataframe(filepath, format)

    def update_data(self, symbol=None, levels=None):
        """
        Update data for specified symbol and levels.

        Parameters:
        -----------
        symbol : str, optional
            Trading pair to update (if None, updates all)
        levels : list, optional
            Levels to update (if None, updates all)

        Returns:
        --------
        dict
            Dictionary with updated data
        """
        symbols_to_update = [symbol] if symbol else self.symbols
        levels_to_update = levels or list(self.timeframes.keys())

        for sym in symbols_to_update:
            if sym not in self.data:
                self.data[sym] = {}

            for level in levels_to_update:
                timeframe = self.timeframes.get(level)

                if not timeframe:
                    logger.warning(f"No timeframe configured for level: {level}")
                    continue

                # Get most recent data from cache
                cached_data = self.load_cached_data(sym, level)

                if cached_data is not None and not cached_data.empty:
                    # Get last timestamp
                    last_timestamp = cached_data.index[-1]

                    # Fetch only newer data
                    new_data = self.fetch_ohlcv_data(
                        sym,
                        timeframe,
                        since=last_timestamp,
                        limit=1000
                    )

                    if new_data is not None and not new_data.empty:
                        # Remove overlapping data
                        new_data = new_data[new_data.index > last_timestamp]

                        if not new_data.empty:
                            # Combine old and new data
                            updated_data = pd.concat([cached_data, new_data])
                            self.data[sym][level] = updated_data

                            # Save updated data
                            self.save_data(sym, level, updated_data)

                            logger.info(f"Updated {sym} {level} data with {len(new_data)} new candles")
                        else:
                            # No new data, use cached
                            self.data[sym][level] = cached_data
                            logger.info(f"No new data for {sym} {level}, using cached data")
                    else:
                        # Error fetching new data, use cached
                        self.data[sym][level] = cached_data
                        logger.warning(f"Error fetching new data for {sym} {level}, using cached data")
                else:
                    # No cached data, fetch fresh
                    lookback_days = 30 if level == 'macro' else 7 if level == 'meso' else 1
                    fresh_data = self.fetch_recent_data(sym, timeframe, lookback_days)

                    if fresh_data is not None and not fresh_data.empty:
                        self.data[sym][level] = fresh_data
                        self.save_data(sym, level, fresh_data)
                        logger.info(f"Fetched fresh data for {sym} {level} with {len(fresh_data)} candles")
                    else:
                        logger.error(f"Failed to fetch data for {sym} {level}")

        return self.data

    def get_data(self, symbol, level):
        """
        Get data for a specific symbol and level.

        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., 'BTC/USDT')
        level : str
            Volatility level ('micro', 'meso', 'macro')

        Returns:
        --------
        pd.DataFrame
            OHLCV data
        """
        if symbol in self.data and level in self.data[symbol]:
            return self.data[symbol][level]

        # Try to load from cache
        cached_data = self.load_cached_data(symbol, level)

        if cached_data is not None and not cached_data.empty:
            if symbol not in self.data:
                self.data[symbol] = {}

            self.data[symbol][level] = cached_data
            return cached_data

        # No data available, fetch it
        logger.info(f"No data available for {symbol} {level}, fetching...")
        self.update_data(symbol, [level])

        if symbol in self.data and level in self.data[symbol]:
            return self.data[symbol][level]

        return None

# Factory function to get a configured collector
def get_exchange_data_collector():
    """
    Get a configured exchange data collector.

    Returns:
    --------
    ExchangeDataCollector
        Configured collector instance
    """
    return ExchangeDataCollector(
        exchange_id=config.get('exchange.name', 'binance'),
        api_key=config.get('exchange.api_key', ''),
        api_secret=config.get('exchange.api_secret', ''),
        symbols=config.get('symbols', ['BTC/USDT']),
        timeframes={
            'micro': config.get('timeframes.micro', '1m'),
            'meso': config.get('timeframes.meso', '1h'),
            'macro': config.get('timeframes.macro', '1d')
        }
    )