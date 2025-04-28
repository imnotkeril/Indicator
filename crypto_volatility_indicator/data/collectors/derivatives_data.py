"""
Module for collecting data from derivatives markets.
Handles connection to exchanges and fetches futures, options, and funding rate data.
"""
import os
import sys
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.config import config
from crypto_volatility_indicator.utils.logger import get_data_logger
from crypto_volatility_indicator.utils.helpers import (
    ensure_directory,
    save_dataframe,
    load_dataframe,
    timestamp_to_datetime
)

# Set up logger
logger = get_data_logger()


class DerivativesDataCollector:
    """
    Collector for derivatives market data.

    This class handles connections to cryptocurrency derivatives exchanges
    and fetches futures, options, and funding rate data.
    """

    def __init__(self, exchange_id=None, api_key=None, api_secret=None, symbols=None):
        """
        Initialize the derivatives data collector.

        Parameters:
        -----------
        exchange_id : str, optional
            Exchange ID (e.g., 'deribit', 'binance-futures')
            If None, uses the value from config
        api_key : str, optional
            API key for the exchange
            If None, uses the value from config
        api_secret : str, optional
            API secret for the exchange
            If None, uses the value from config
        symbols : list, optional
            List of trading pairs to collect data for (e.g., ['BTC/USD', 'ETH/USD'])
            If None, uses the value from config
        """
        # Load configuration if not provided
        self.exchange_id = exchange_id or config.get('derivatives.exchange.name', 'deribit')
        self.api_key = api_key or config.get('derivatives.exchange.api_key', '')
        self.api_secret = api_secret or config.get('derivatives.exchange.api_secret', '')

        # Map config symbols to derivatives symbols if needed
        spot_symbols = symbols or config.get('symbols', ['BTC/USDT'])
        self.symbols = self._map_spot_to_derivatives_symbols(spot_symbols)

        # Initialize exchange
        self.initialize_exchange()

        # Data storage
        self.futures_data = {}
        self.options_data = {}
        self.funding_rates = {}
        self.open_interest = {}

        # Cache directory
        self.cache_dir = Path('data/cache/derivatives')
        ensure_directory(self.cache_dir)

        logger.info(f"DerivativesDataCollector initialized for {self.exchange_id} with symbols: {self.symbols}")

    def _map_spot_to_derivatives_symbols(self, spot_symbols):
        """
        Map spot market symbols to derivatives market symbols.

        Parameters:
        -----------
        spot_symbols : list
            List of spot market symbols (e.g., ['BTC/USDT', 'ETH/USDT'])

        Returns:
        --------
        list
            List of derivatives market symbols
        """
        derivatives_symbols = []

        for symbol in spot_symbols:
            base, quote = symbol.split('/')

            if self.exchange_id == 'deribit':
                derivatives_symbols.append(f"{base}-PERPETUAL")  # Deribit uses this format
            elif self.exchange_id.startswith('binance'):
                derivatives_symbols.append(f"{base}/USDT:USDT")  # Binance Futures format
            elif self.exchange_id == 'bitmex':
                if base == 'BTC':
                    derivatives_symbols.append('XBTUSD')  # BitMEX uses XBT for Bitcoin
                else:
                    derivatives_symbols.append(f"{base}USD")
            else:
                # Default to same as spot
                derivatives_symbols.append(symbol)

        return derivatives_symbols

    def initialize_exchange(self):
        """Initialize the connection to the exchange."""
        try:
            # Handle special case for Binance Futures
            if self.exchange_id == 'binance-futures':
                exchange_class = getattr(ccxt, 'binance')
            else:
                exchange_class = getattr(ccxt, self.exchange_id)

            # Create exchange instance
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'timeout': config.get('derivatives.exchange.timeout', 30000)
            })

            # Set up any exchange-specific settings
            if self.exchange_id == 'binance-futures':
                self.exchange.options['defaultType'] = 'future'
            elif self.exchange_id == 'deribit':
                self.exchange.options['adjustForTimeDifference'] = True

            logger.info(f"Connected to {self.exchange_id} exchange")

        except Exception as e:
            logger.error(f"Failed to initialize exchange {self.exchange_id}: {e}")
            raise

    def fetch_futures_contracts(self, symbol=None):
        """
        Fetch futures contracts for a symbol or all symbols.

        Parameters:
        -----------
        symbol : str, optional
            Symbol to fetch contracts for (if None, fetches for all symbols)

        Returns:
        --------
        dict
            Dictionary of futures contracts by symbol
        """
        symbols_to_fetch = [symbol] if symbol else self.symbols
        futures_contracts = {}

        for sym in symbols_to_fetch:
            try:
                logger.info(f"Fetching futures contracts for {sym}")

                # Load markets if needed
                if not self.exchange.markets:
                    self.exchange.load_markets()

                # Get base symbol (e.g., 'BTC' from 'BTC-PERPETUAL')
                base_symbol = sym.split('-')[0] if '-' in sym else sym.split('/')[0]

                # Find all futures for this base symbol
                contracts = []

                for market_id, market in self.exchange.markets.items():
                    # Check if it's a future and matches the base symbol
                    is_future = market.get('type') == 'future' or (market.get('future') and market.get('active', True))

                    if is_future and base_symbol in market_id.split('/')[0]:
                        contracts.append(market)

                if not contracts:
                    logger.warning(f"No futures contracts found for {sym}")
                    continue

                # Fetch ticker data for each contract
                contracts_data = []

                for contract in contracts:
                    try:
                        contract_id = contract['id']
                        ticker = self.exchange.fetch_ticker(contract_id)

                        # Calculate expiry date if available
                        expiry = None

                        if 'expiry' in contract and contract['expiry']:
                            expiry_timestamp = contract['expiry'] / 1000 if isinstance(contract['expiry'],
                                                                                       int) else None
                            if expiry_timestamp:
                                expiry = timestamp_to_datetime(expiry_timestamp, 's')

                        contracts_data.append({
                            'symbol': contract_id,
                            'base_symbol': base_symbol,
                            'type': 'perpetual' if 'PERP' in contract_id.upper() or 'PERPETUAL' in contract_id.upper() else 'future',
                            'expiry': expiry.strftime('%Y-%m-%d') if expiry else 'perpetual',
                            'last_price': ticker.get('last', None),
                            'mark_price': ticker.get('mark', ticker.get('last', None)),
                            'bid': ticker.get('bid', None),
                            'ask': ticker.get('ask', None),
                            'volume_24h': ticker.get('volume', None),
                            'open_interest': ticker.get('openInterest', None),
                            'timestamp': ticker.get('timestamp', int(time.time() * 1000))
                        })
                    except Exception as e:
                        logger.error(f"Error fetching ticker for {contract['id']}: {e}")

                # Convert to DataFrame
                if contracts_data:
                    df = pd.DataFrame(contracts_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    futures_contracts[base_symbol] = df
                    logger.info(f"Fetched {len(df)} futures contracts for {base_symbol}")
                else:
                    logger.warning(f"Failed to fetch futures contracts data for {sym}")

            except Exception as e:
                logger.error(f"Error fetching futures contracts for {sym}: {e}")

        self.futures_data = futures_contracts
        return futures_contracts

    def fetch_options_chain(self, symbol=None):
        """
        Fetch options chain for a symbol or all symbols.

        Parameters:
        -----------
        symbol : str, optional
            Symbol to fetch options for (if None, fetches for all symbols)

        Returns:
        --------
        dict
            Dictionary of options contracts by symbol
        """
        # Check if exchange supports options
        if not hasattr(self.exchange, 'fetchOptionChain') and self.exchange_id != 'deribit':
            logger.warning(f"Exchange {self.exchange_id} does not support options fetching")
            return {}

        symbols_to_fetch = [symbol] if symbol else self.symbols
        options_chains = {}

        for sym in symbols_to_fetch:
            try:
                logger.info(f"Fetching options chain for {sym}")

                # Get base symbol (e.g., 'BTC' from 'BTC-PERPETUAL')
                base_symbol = sym.split('-')[0] if '-' in sym else sym.split('/')[0]

                # Fetch options
                options_data = []

                # Deribit-specific implementation
                if self.exchange_id == 'deribit':
                    # Load markets if needed
                    if not self.exchange.markets:
                        self.exchange.load_markets()

                    # Get all instruments for this currency
                    instruments = self.exchange.public_get_instruments({
                        'currency': base_symbol,
                        'kind': 'option',
                        'expired': False
                    })

                    if not instruments or 'result' not in instruments:
                        logger.warning(f"No options found for {base_symbol}")
                        continue

                    for instrument in instruments['result']:
                        # Parse instrument data
                        instrument_name = instrument['instrument_name']
                        option_type = 'call' if instrument_name.endswith('C') else 'put'
                        strike_price = float(instrument_name.split('-')[-2])
                        expiry_date = datetime.strptime(instrument['expiration_timestamp'][:10], '%Y-%m-%d')

                        # Fetch ticker
                        try:
                            ticker = self.exchange.public_get_ticker({
                                'instrument_name': instrument_name
                            })

                            if ticker and 'result' in ticker:
                                ticker_data = ticker['result']

                                options_data.append({
                                    'symbol': instrument_name,
                                    'base_symbol': base_symbol,
                                    'option_type': option_type,
                                    'strike': strike_price,
                                    'expiry': expiry_date.strftime('%Y-%m-%d'),
                                    'last_price': ticker_data.get('last_price', None),
                                    'mark_price': ticker_data.get('mark_price', None),
                                    'underlying_price': ticker_data.get('underlying_price', None),
                                    'bid': ticker_data.get('best_bid_price', None),
                                    'ask': ticker_data.get('best_ask_price', None),
                                    'volume_24h': ticker_data.get('stats', {}).get('volume', None),
                                    'open_interest': ticker_data.get('open_interest', None),
                                    'implied_volatility': ticker_data.get('mark_iv', None),
                                    'timestamp': int(time.time() * 1000)
                                })
                        except Exception as e:
                            logger.error(f"Error fetching ticker for {instrument_name}: {e}")
                # Other exchanges - implement as needed
                else:
                    logger.warning(f"Options fetching not implemented for {self.exchange_id}")

                # Convert to DataFrame
                if options_data:
                    df = pd.DataFrame(options_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    options_chains[base_symbol] = df
                    logger.info(f"Fetched {len(df)} options contracts for {base_symbol}")
                else:
                    logger.warning(f"Failed to fetch options data for {sym}")

            except Exception as e:
                logger.error(f"Error fetching options chain for {sym}: {e}")

        self.options_data = options_chains
        return options_chains

    def fetch_funding_rates(self, symbol=None):
        """
        Fetch funding rates for a symbol or all symbols.

        Parameters:
        -----------
        symbol : str, optional
            Symbol to fetch funding rates for (if None, fetches for all symbols)

        Returns:
        --------
        dict
            Dictionary of funding rates by symbol
        """
        # Check if exchange supports funding rates
        if not hasattr(self.exchange, 'fetchFundingRates'):
            logger.warning(f"Exchange {self.exchange_id} does not support funding rates fetching")
            return {}

        symbols_to_fetch = [symbol] if symbol else self.symbols
        funding_rates_data = {}

        for sym in symbols_to_fetch:
            try:
                logger.info(f"Fetching funding rates for {sym}")

                # Get base symbol (e.g., 'BTC' from 'BTC-PERPETUAL')
                base_symbol = sym.split('-')[0] if '-' in sym else sym.split('/')[0]

                # Fetch funding rate
                funding_rates = self.exchange.fetch_funding_rates([sym])

                if not funding_rates or sym not in funding_rates:
                    logger.warning(f"No funding rates found for {sym}")
                    continue

                # Get the funding rate data
                rate_data = funding_rates[sym]

                # Format the data
                funding_data = {
                    'symbol': sym,
                    'base_symbol': base_symbol,
                    'funding_rate': rate_data.get('rate', None),
                    'funding_time': rate_data.get('timestamp', int(time.time() * 1000)),
                    'next_funding_time': rate_data.get('nextFundingTime', None),
                    'next_funding_rate': rate_data.get('nextFundingRate', None)
                }

                # Convert to DataFrame
                df = pd.DataFrame([funding_data])
                df['timestamp'] = pd.to_datetime(df['funding_time'], unit='ms')
                df.set_index('timestamp', inplace=True)

                funding_rates_data[base_symbol] = df
                logger.info(f"Fetched funding rates for {base_symbol}")

            except Exception as e:
                logger.error(f"Error fetching funding rates for {sym}: {e}")

            self.funding_rates = funding_rates_data
            return funding_rates_data

    def fetch_historical_funding_rates(self, symbol, days=30):
        """
        Fetch historical funding rates for a symbol.

        Parameters:
        -----------
        symbol : str
            Symbol to fetch historical funding rates for
        days : int, optional
            Number of days of historical data to fetch

        Returns:
        --------
        pd.DataFrame
            DataFrame with historical funding rates
        """
        # Check if exchange supports historical funding rates
        if not hasattr(self.exchange, 'fetchFundingRateHistory'):
            logger.warning(f"Exchange {self.exchange_id} does not support historical funding rates fetching")
            return None

        try:
            logger.info(f"Fetching historical funding rates for {symbol}")

            # Calculate start time
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Fetch historical funding rates
            funding_history = self.exchange.fetch_funding_rate_history(symbol, since)

            if not funding_history:
                logger.warning(f"No historical funding rates found for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(funding_history)

            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)

            logger.info(f"Fetched {len(df)} historical funding rates for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical funding rates for {symbol}: {e}")
            return None

    def fetch_open_interest(self, symbol=None):
        """
        Fetch open interest for a symbol or all symbols.

        Parameters:
        -----------
        symbol : str, optional
            Symbol to fetch open interest for (if None, fetches for all symbols)

        Returns:
        --------
        dict
            Dictionary of open interest by symbol
        """
        # Check if exchange supports open interest
        if not hasattr(self.exchange, 'fetchOpenInterest') and self.exchange_id != 'deribit':
            logger.warning(f"Exchange {self.exchange_id} does not support open interest fetching")
            return {}

        symbols_to_fetch = [symbol] if symbol else self.symbols
        open_interest_data = {}

        for sym in symbols_to_fetch:
            try:
                logger.info(f"Fetching open interest for {sym}")

                # Get base symbol (e.g., 'BTC' from 'BTC-PERPETUAL')
                base_symbol = sym.split('-')[0] if '-' in sym else sym.split('/')[0]

                # Fetch open interest
                oi_data = None

                # Exchange-specific implementations
                if self.exchange_id == 'deribit':
                    # For Deribit, fetch open interest by currency
                    response = self.exchange.public_get_book_summary_by_currency({
                        'currency': base_symbol
                    })

                    if response and 'result' in response:
                        # Parse the response
                        oi_records = []

                        for item in response['result']:
                            if 'instrument_name' in item and 'open_interest' in item:
                                instrument = item['instrument_name']

                                # Determine instrument type
                                if 'PERPETUAL' in instrument:
                                    instrument_type = 'perpetual'
                                elif instrument.endswith('C') or instrument.endswith('P'):
                                    instrument_type = 'option'
                                else:
                                    instrument_type = 'future'

                                oi_records.append({
                                    'symbol': instrument,
                                    'base_symbol': base_symbol,
                                    'type': instrument_type,
                                    'open_interest': item['open_interest'],
                                    'open_interest_usd': item.get('open_interest_usd', None),
                                    'volume_24h': item.get('volume_24h', None),
                                    'timestamp': int(time.time() * 1000)
                                })

                        if oi_records:
                            oi_data = pd.DataFrame(oi_records)
                            oi_data['timestamp'] = pd.to_datetime(oi_data['timestamp'], unit='ms')
                            oi_data.set_index('timestamp', inplace=True)
                # Other exchanges - implement as needed
                elif hasattr(self.exchange, 'fetchOpenInterest'):
                    oi = self.exchange.fetch_open_interest(sym)

                    if oi:
                        oi_records = [{
                            'symbol': sym,
                            'base_symbol': base_symbol,
                            'type': 'perpetual' if 'PERP' in sym.upper() or 'PERPETUAL' in sym.upper() else 'future',
                            'open_interest': oi.get('openInterest', 0),
                            'open_interest_usd': oi.get('openInterestUsd', None),
                            'timestamp': oi.get('timestamp', int(time.time() * 1000))
                        }]

                        oi_data = pd.DataFrame(oi_records)
                        oi_data['timestamp'] = pd.to_datetime(oi_data['timestamp'], unit='ms')
                        oi_data.set_index('timestamp', inplace=True)

                if oi_data is not None:
                    open_interest_data[base_symbol] = oi_data
                    logger.info(f"Fetched open interest for {base_symbol} ({len(oi_data)} instruments)")
                else:
                    logger.warning(f"Failed to fetch open interest for {sym}")

            except Exception as e:
                logger.error(f"Error fetching open interest for {sym}: {e}")

        self.open_interest = open_interest_data
        return open_interest_data

    def calculate_implied_volatility_index(self, asset):
        """
        Calculate a VIX-like implied volatility index for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
        --------
        float
            Implied volatility index value
        """
        if asset not in self.options_data:
            logger.warning(f"No options data available for {asset}")
            return None

        options_df = self.options_data[asset]

        if options_df.empty:
            logger.warning(f"Empty options data for {asset}")
            return None

        try:
            # Filter options with 23-37 days to expiry (close to 30 days)
            today = datetime.now().date()

            # Convert expiry to datetime
            options_df['expiry_date'] = pd.to_datetime(options_df['expiry']).dt.date

            # Calculate days to expiry
            options_df['days_to_expiry'] = (options_df['expiry_date'] - today).dt.days

            # Filter near-term options
            near_term = options_df[
                (options_df['days_to_expiry'] >= 23) &
                (options_df['days_to_expiry'] <= 37)
                ]

            if near_term.empty:
                # If no options in ideal range, use closest available
                median_days = options_df['days_to_expiry'].median()
                near_term = options_df[
                    (options_df['days_to_expiry'] >= median_days * 0.8) &
                    (options_df['days_to_expiry'] <= median_days * 1.2)
                    ]

            if near_term.empty:
                logger.warning(f"No suitable options found for IV index calculation ({asset})")
                return None

            # Use implied volatility if available
            if 'implied_volatility' in near_term.columns and not near_term['implied_volatility'].isna().all():
                # Calculate ATM options
                # Find the current underlying price
                underlying_price = near_term['underlying_price'].iloc[0]

                # Calculate moneyness (distance from ATM)
                near_term['moneyness'] = abs(near_term['strike'] / underlying_price - 1)

                # Get near-the-money options (lowest moneyness)
                atm_options = near_term.nsmallest(5, 'moneyness')

                # Calculate weighted average IV
                if not atm_options.empty and not atm_options['implied_volatility'].isna().all():
                    # Weight by inverse of moneyness (closer to ATM = higher weight)
                    atm_options['weight'] = 1 / (atm_options['moneyness'] + 0.01)
                    weighted_iv = (atm_options['implied_volatility'] * atm_options['weight']).sum() / atm_options[
                        'weight'].sum()

                    # Convert to percentage points
                    iv_index = weighted_iv * 100

                    logger.info(f"Calculated IV index for {asset}: {iv_index:.2f}")
                    return iv_index

            # If no implied volatility data available
            logger.warning(f"No implied volatility data available for {asset}")
            return None

        except Exception as e:
            logger.error(f"Error calculating IV index for {asset}: {e}")
            return None

    def calculate_put_call_ratio(self, asset):
        """
        Calculate put-call ratio for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
        --------
        float
            Put-call ratio
        """
        if asset not in self.options_data:
            logger.warning(f"No options data available for {asset}")
            return None

        options_df = self.options_data[asset]

        if options_df.empty:
            logger.warning(f"Empty options data for {asset}")
            return None

        try:
            # Count puts and calls
            puts = options_df[options_df['option_type'] == 'put']
            calls = options_df[options_df['option_type'] == 'call']

            if len(calls) == 0:
                logger.warning(f"No call options found for {asset}")
                return None

            # Calculate put-call ratio
            put_call_ratio = len(puts) / len(calls)

            logger.info(f"Calculated put-call ratio for {asset}: {put_call_ratio:.2f}")
            return put_call_ratio

        except Exception as e:
            logger.error(f"Error calculating put-call ratio for {asset}: {e}")
            return None

    def calculate_term_structure(self, asset):
        """
        Calculate the volatility term structure for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility term structure
        """
        if asset not in self.options_data:
            logger.warning(f"No options data available for {asset}")
            return None

        options_df = self.options_data[asset]

        if options_df.empty:
            logger.warning(f"Empty options data for {asset}")
            return None

        try:
            # Convert expiry to datetime
            options_df['expiry_date'] = pd.to_datetime(options_df['expiry']).dt.date

            # Calculate days to expiry
            today = datetime.now().date()
            options_df['days_to_expiry'] = (options_df['expiry_date'] - today).dt.days

            # Group by days to expiry
            term_structure = []

            for days, group in options_df.groupby('days_to_expiry'):
                if len(group) < 2:
                    continue

                # Get underlying price
                underlying_price = group['underlying_price'].iloc[0]

                # Calculate moneyness
                group['moneyness'] = abs(group['strike'] / underlying_price - 1)

                # Find near-ATM options (smallest absolute moneyness)
                near_atm = group.nsmallest(3, 'moneyness')

                if len(near_atm) > 0 and 'implied_volatility' in near_atm.columns:
                    # Calculate average IV of near-ATM options
                    atm_iv = near_atm['implied_volatility'].mean()

                    term_structure.append({
                        'days_to_expiry': days,
                        'atm_implied_volatility': atm_iv
                    })

            if not term_structure:
                logger.warning(f"Could not calculate term structure for {asset}")
                return None

            # Create DataFrame and sort by days to expiry
            term_df = pd.DataFrame(term_structure).sort_values('days_to_expiry')

            logger.info(f"Calculated volatility term structure for {asset} with {len(term_df)} terms")
            return term_df

        except Exception as e:
            logger.error(f"Error calculating term structure for {asset}: {e}")
            return None

    def calculate_futures_basis(self, asset):
        """
        Calculate futures basis (premium/discount) for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
        --------
        pd.DataFrame
            DataFrame with futures basis for different expirations
        """
        if asset not in self.futures_data:
            logger.warning(f"No futures data available for {asset}")
            return None

        futures_df = self.futures_data[asset]

        if futures_df.empty:
            logger.warning(f"Empty futures data for {asset}")
            return None

        try:
            # Find perpetual contract for reference price
            perpetual = futures_df[futures_df['type'] == 'perpetual']

            if perpetual.empty:
                logger.warning(f"No perpetual contract found for {asset}")
                return None

            # Get reference price
            reference_price = perpetual['mark_price'].iloc[0]

            # Calculate basis for each future
            futures_df['basis_pct'] = (futures_df['mark_price'] / reference_price - 1) * 100

            # For expired futures, calculate days to expiry
            if 'expiry' in futures_df.columns and futures_df['expiry'].dtype == 'object':
                # Filter out perpetual contracts
                expiring_futures = futures_df[futures_df['expiry'] != 'perpetual']

                if not expiring_futures.empty:
                    # Convert expiry to datetime
                    expiring_futures['expiry_date'] = pd.to_datetime(expiring_futures['expiry']).dt.date

                    # Calculate days to expiry
                    today = datetime.now().date()
                    expiring_futures['days_to_expiry'] = (expiring_futures['expiry_date'] - today).dt.days

                    # Calculate annualized basis
                    expiring_futures['annualized_basis'] = expiring_futures['basis_pct'] * 365 / expiring_futures[
                        'days_to_expiry']

                    # Combine with perpetual
                    result = pd.concat([perpetual, expiring_futures])
                else:
                    result = futures_df
            else:
                result = futures_df

            logger.info(f"Calculated futures basis for {asset}")
            return result

        except Exception as e:
            logger.error(f"Error calculating futures basis for {asset}: {e}")
            return None

    def save_data(self, data_type, asset, data=None, format='parquet'):
        """
        Save data to cache.

        Parameters:
        -----------
        data_type : str
            Type of data ('futures', 'options', 'funding_rates', 'open_interest')
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')
        data : pd.DataFrame, optional
            Data to save (if None, uses the corresponding instance attribute)
        format : str, optional
            File format ('csv', 'pickle', 'parquet')

        Returns:
        --------
        Path
            Path to saved file
        """
        # Use provided data or get from instance
        if data is None:
            if data_type == 'futures':
                data = self.futures_data.get(asset)
            elif data_type == 'options':
                data = self.options_data.get(asset)
            elif data_type == 'funding_rates':
                data = self.funding_rates.get(asset)
            elif data_type == 'open_interest':
                data = self.open_interest.get(asset)
            else:
                logger.error(f"Unknown data type: {data_type}")
                return None

        if data is None or data.empty:
            logger.warning(f"No {data_type} data available for {asset}")
            return None

        # Create asset-specific directory
        asset_dir = self.cache_dir / asset.lower()
        ensure_directory(asset_dir)

        # Current date string
        date_str = datetime.now().strftime('%Y%m%d')

        # Create filename
        filename = f"{asset.lower()}_{data_type}_{date_str}.{format}"
        filepath = asset_dir / filename

        # Save data
        try:
            save_dataframe(data, filepath, format)
            logger.info(f"Saved {data_type} data for {asset} to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving {data_type} data for {asset}: {e}")
            return None

    def load_data(self, data_type, asset, date=None, format='parquet'):
        """
        Load data from cache.

        Parameters:
        -----------
        data_type : str
            Type of data ('futures', 'options', 'funding_rates', 'open_interest')
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')
        date : datetime or str, optional
            Date to load data for (if None, loads most recent)
        format : str, optional
            File format ('csv', 'pickle', 'parquet')

        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        # Create asset-specific directory path
        asset_dir = self.cache_dir / asset.lower()

        if not asset_dir.exists():
            logger.warning(f"No cached data found for {asset}")
            return None

        # Format date if it's a datetime
        if isinstance(date, datetime):
            date_str = date.strftime('%Y%m%d')
        else:
            date_str = date

        # Find most recent file if date not specified
        if date_str:
            pattern = f"{asset.lower()}_{data_type}_{date_str}.{format}"
        else:
            pattern = f"{asset.lower()}_{data_type}_*.{format}"

        files = list(asset_dir.glob(pattern))

        if not files:
            logger.warning(f"No cached {data_type} data found for {asset}")
            return None

        # Get most recent file if multiple match
        if len(files) > 1 and date_str is None:
            files.sort(key=lambda p: p.stem.split('_')[-1], reverse=True)

        # Load data
        try:
            df = load_dataframe(files[0], format)
            logger.info(f"Loaded {data_type} data for {asset} from {files[0]}")
            return df
        except Exception as e:
            logger.error(f"Error loading {data_type} data for {asset}: {e}")
            return None

    def update_all_data(self, asset=None):
        """
        Update all data for an asset or all assets.

        Parameters:
        -----------
        asset : str, optional
            Asset to update data for (if None, updates all)

        Returns:
        --------
        dict
            Dictionary with updated data
        """
        assets_to_update = [asset] if asset else [s.split('-')[0] if '-' in s else s.split('/')[0] for s in
                                                  self.symbols]

        updated_data = {
            'futures': {},
            'options': {},
            'funding_rates': {},
            'open_interest': {}
        }

        for ast in assets_to_update:
            logger.info(f"Updating all derivatives data for {ast}")

            # Fetch futures
            futures = self.fetch_futures_contracts(ast)
            if ast in futures:
                updated_data['futures'][ast] = futures[ast]
                self.save_data('futures', ast, futures[ast])

            # Fetch options
            options = self.fetch_options_chain(ast)
            if ast in options:
                updated_data['options'][ast] = options[ast]
                self.save_data('options', ast, options[ast])

            # Fetch funding rates
            funding = self.fetch_funding_rates(ast)
            if ast in funding:
                updated_data['funding_rates'][ast] = funding[ast]
                self.save_data('funding_rates', ast, funding[ast])

            # Fetch open interest
            oi = self.fetch_open_interest(ast)
            if ast in oi:
                updated_data['open_interest'][ast] = oi[ast]
                self.save_data('open_interest', ast, oi[ast])

        return updated_data

    def get_implied_volatility_metrics(self, asset):
        """
        Get comprehensive implied volatility metrics for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
        --------
        dict
            Dictionary with volatility metrics
        """
        # Make sure we have options data
        if asset not in self.options_data or self.options_data[asset].empty:
            # Try to load from cache
            cached_options = self.load_data('options', asset)

            if cached_options is not None and not cached_options.empty:
                self.options_data[asset] = cached_options
            else:
                # Fetch new data
                self.fetch_options_chain(asset)

        if asset not in self.options_data or self.options_data[asset].empty:
            logger.warning(f"No options data available for {asset}")
            return None

        metrics = {
            'asset': asset,
            'timestamp': datetime.now().isoformat()
        }

        # Calculate IV index
        iv_index = self.calculate_implied_volatility_index(asset)
        if iv_index is not None:
            metrics['iv_index'] = iv_index

        # Calculate put-call ratio
        pc_ratio = self.calculate_put_call_ratio(asset)
        if pc_ratio is not None:
            metrics['put_call_ratio'] = pc_ratio

        # Calculate term structure
        term_structure = self.calculate_term_structure(asset)
        if term_structure is not None:
            metrics['term_structure'] = term_structure

        # Funding rates (for perpetual futures)
        if asset in self.funding_rates and not self.funding_rates[asset].empty:
            metrics['funding_rate'] = self.funding_rates[asset]['funding_rate'].iloc[0]

        # Futures basis
        basis = self.calculate_futures_basis(asset)
        if basis is not None:
            # Aggregate basis metrics
            non_perpetual = basis[basis['type'] != 'perpetual']

            if not non_perpetual.empty:
                metrics['futures_basis'] = non_perpetual['basis_pct'].mean()

                if 'annualized_basis' in non_perpetual.columns:
                    metrics['annualized_basis'] = non_perpetual['annualized_basis'].mean()

        # Add interpretation
        metrics['market_sentiment'] = self._interpret_volatility_metrics(metrics)

        return metrics

    def _interpret_volatility_metrics(self, metrics):
        """
        Interpret volatility metrics to determine market sentiment.

        Parameters:
        -----------
        metrics : dict
            Dictionary with volatility metrics

        Returns:
        --------
        dict
            Dictionary with market sentiment interpretation
        """
        sentiment = {
            'overall': 'neutral',
            'factors': {}
        }

        # Interpret IV index
        if 'iv_index' in metrics:
            iv = metrics['iv_index']

            if iv > 100:
                sentiment['factors']['iv_index'] = {
                    'value': iv,
                    'interpretation': 'bearish',
                    'description': 'Very high implied volatility indicates market fear'
                }
            elif iv > 80:
                sentiment['factors']['iv_index'] = {
                    'value': iv,
                    'interpretation': 'slightly_bearish',
                    'description': 'Elevated implied volatility indicates caution'
                }
            elif iv < 40:
                sentiment['factors']['iv_index'] = {
                    'value': iv,
                    'interpretation': 'bullish',
                    'description': 'Low implied volatility indicates market complacency'
                }
            else:
                sentiment['factors']['iv_index'] = {
                    'value': iv,
                    'interpretation': 'neutral',
                    'description': 'Moderate implied volatility'
                }

        # Interpret put-call ratio
        if 'put_call_ratio' in metrics:
            pc_ratio = metrics['put_call_ratio']

            if pc_ratio > 1.5:
                sentiment['factors']['put_call_ratio'] = {
                    'value': pc_ratio,
                    'interpretation': 'bearish',
                    'description': 'High put-call ratio indicates bearish sentiment'
                }
            elif pc_ratio > 1.2:
                sentiment['factors']['put_call_ratio'] = {
                    'value': pc_ratio,
                    'interpretation': 'slightly_bearish',
                    'description': 'Elevated put-call ratio'
                }
            elif pc_ratio < 0.5:
                sentiment['factors']['put_call_ratio'] = {
                    'value': pc_ratio,
                    'interpretation': 'bullish',
                    'description': 'Low put-call ratio indicates bullish sentiment'
                }
            elif pc_ratio < 0.8:
                sentiment['factors']['put_call_ratio'] = {
                    'value': pc_ratio,
                    'interpretation': 'slightly_bullish',
                    'description': 'Low put-call ratio'
                }
            else:
                sentiment['factors']['put_call_ratio'] = {
                    'value': pc_ratio,
                    'interpretation': 'neutral',
                    'description': 'Balanced put-call ratio'
                }

        # Interpret term structure
        if 'term_structure' in metrics and not metrics['term_structure'].empty:
            ts = metrics['term_structure']

            if len(ts) >= 2:
                # Calculate slope
                x = np.log(ts['days_to_expiry'].values)
                y = ts['atm_implied_volatility'].values

                # Linear regression
                slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)[0:5]

                # Interpret
                if slope > 0.05:
                    sentiment['factors']['term_structure'] = {
                        'value': slope,
                        'interpretation': 'slightly_bullish',
                        'description': 'Upward sloping volatility term structure (contango)'
                    }
                elif slope > 0.1:
                    sentiment['factors']['term_structure'] = {
                        'value': slope,
                        'interpretation': 'bullish',
                        'description': 'Steep upward sloping volatility term structure'
                    }
                elif slope < -0.1:
                    sentiment['factors']['term_structure'] = {
                        'value': slope,
                        'interpretation': 'bearish',
                        'description': 'Downward sloping volatility term structure (backwardation)'
                    }
                elif slope < -0.05:
                    sentiment['factors']['term_structure'] = {
                        'value': slope,
                        'interpretation': 'slightly_bearish',
                        'description': 'Slightly downward sloping volatility term structure'
                    }
                else:
                    sentiment['factors']['term_structure'] = {
                        'value': slope,
                        'interpretation': 'neutral',
                        'description': 'Flat volatility term structure'
                    }

        # Interpret funding rate
        if 'funding_rate' in metrics:
            fr = metrics['funding_rate']

            if fr > 0.01:  # 1% (usually annualized)
                sentiment['factors']['funding_rate'] = {
                    'value': fr,
                    'interpretation': 'bullish',
                    'description': 'High positive funding rate indicates bullish sentiment'
                }
            elif fr > 0.005:  # 0.5%
                sentiment['factors']['funding_rate'] = {
                    'value': fr,
                    'interpretation': 'slightly_bullish',
                    'description': 'Positive funding rate'
                }
            elif fr < -0.01:  # -1%
                sentiment['factors']['funding_rate'] = {
                    'value': fr,
                    'interpretation': 'bearish',
                    'description': 'Negative funding rate indicates bearish sentiment'
                }
            elif fr < -0.005:  # -0.5%
                sentiment['factors']['funding_rate'] = {
                    'value': fr,
                    'interpretation': 'slightly_bearish',
                    'description': 'Negative funding rate'
                }
            else:
                sentiment['factors']['funding_rate'] = {
                    'value': fr,
                    'interpretation': 'neutral',
                    'description': 'Neutral funding rate'
                }

            # Interpret futures basis
        if 'futures_basis' in metrics:
            basis = metrics['futures_basis']

            if basis > 10:  # 10%
                sentiment['factors']['futures_basis'] = {
                    'value': basis,
                    'interpretation': 'bullish',
                    'description': 'High futures premium indicates bullish sentiment'
                }
            elif basis > 5:  # 5%
                sentiment['factors']['futures_basis'] = {
                    'value': basis,
                    'interpretation': 'slightly_bullish',
                    'description': 'Positive futures premium'
                }
            elif basis < -5:  # -5%
                sentiment['factors']['futures_basis'] = {
                    'value': basis,
                    'interpretation': 'bearish',
                    'description': 'Futures discount indicates bearish sentiment'
                }
            elif basis < -2:  # -2%
                sentiment['factors']['futures_basis'] = {
                    'value': basis,
                    'interpretation': 'slightly_bearish',
                    'description': 'Futures trading at a discount'
                }
            else:
                sentiment['factors']['futures_basis'] = {
                    'value': basis,
                    'interpretation': 'neutral',
                    'description': 'Neutral futures basis'
                }

            # Determine overall sentiment
        if sentiment['factors']:
            interpretations = [factor['interpretation'] for factor in sentiment['factors'].values()]

            # Count occurrences of each interpretation
            counts = {
                'bullish': interpretations.count('bullish') + 0.5 * interpretations.count('slightly_bullish'),
                'bearish': interpretations.count('bearish') + 0.5 * interpretations.count('slightly_bearish'),
                'neutral': interpretations.count('neutral')
            }

            if counts['bullish'] > counts['bearish'] + counts['neutral']:
                sentiment['overall'] = 'bullish'
            elif counts['bearish'] > counts['bullish'] + counts['neutral']:
                sentiment['overall'] = 'bearish'
            elif counts['bullish'] > counts['bearish']:
                sentiment['overall'] = 'slightly_bullish'
            elif counts['bearish'] > counts['bullish']:
                sentiment['overall'] = 'slightly_bearish'
            else:
                sentiment['overall'] = 'neutral'

        return sentiment

# Factory function to get a configured collector
def get_derivatives_data_collector():
    """
    Get a configured derivatives data collector.

    Returns:
    --------
    DerivativesDataCollector
        Configured collector instance
    """
    return DerivativesDataCollector(
        exchange_id=config.get('derivatives.exchange.name', 'deribit'),
        api_key=config.get('derivatives.exchange.api_key', ''),
        api_secret=config.get('derivatives.exchange.api_secret', ''),
        symbols=config.get('symbols', ['BTC/USDT'])
    )