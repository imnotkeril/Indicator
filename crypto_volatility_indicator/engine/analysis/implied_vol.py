"""
Module for analyzing implied volatility from options data.
Calculates and analyzes implied volatility metrics from cryptocurrency options markets.
"""
import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
import ccxt
import requests
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from crypto_volatility_indicator.utils.logger import get_logger
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# Set up logger
logger = get_logger(__name__)

class ImpliedVolatilityAnalyzer:
    """
    Implied Volatility Analyzer for cryptocurrency markets.

    This module focuses on extracting volatility expectations from cryptocurrency options,
    futures, and other derivatives markets. Since not all cryptocurrencies have liquid options markets,
    it uses multiple proxies and alternative data sources to estimate implied volatility.
    """

    def __init__(self, exchange_id='deribit', symbol='BTC/USD', api_key=None, api_secret=None):
        """
        Initialize the ImpliedVolatilityAnalyzer.

        Parameters:
        -----------
        exchange_id : str
            The ID of the exchange to use ('deribit' recommended for options data)
        symbol : str
            The trading pair to analyze (e.g., 'BTC/USD')
        api_key : str, optional
            API key for accessing exchange API (if required)
        api_secret : str, optional
            API secret for accessing exchange API (if required)
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.base_currency = symbol.split('/')[0]  # e.g., 'BTC' from 'BTC/USD'

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })

            logger.info(f"Connected to {exchange_id} exchange for {symbol}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {e}")
            self.exchange = None

        # Initialize data storage
        self.options_data = None
        self.futures_data = None
        self.funding_rates = None
        self.volatility_surface = None
        self.volatility_term_structure = None
        self.historical_iv = None

        # Risk-free rate (used in option pricing models)
        self.risk_free_rate = 0.02  # Default value, can be updated

    def fetch_options_data(self, min_expiry_days=1, max_expiry_days=90, delta_range=None):
        """
        Fetch options data from the exchange.

        Parameters:
        -----------
        min_expiry_days : int
            Minimum days to expiry for options
        max_expiry_days : int
            Maximum days to expiry for options
        delta_range : tuple, optional
            Range of deltas to include (e.g., (0.2, 0.8))

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # This method requires exchange-specific implementation
            # Here we'll implement a generic approach

            logger.info(f"Fetching options data for {self.base_currency}...")

            # Check if the exchange provides options data
            if self.exchange_id not in ['deribit', 'okex', 'ftx', 'bit.com']:
                # For exchanges without options, use a simulation method
                self.options_data = self._simulate_options_data()
                logger.info("Using simulated options data (exchange does not support options)")
                return True

            # For exchanges with options, fetch real data
            # This is a placeholder for the actual API call

            # Calculate date range for options expiries
            now = datetime.now()
            min_date = now + timedelta(days=min_expiry_days)
            max_date = now + timedelta(days=max_expiry_days)

            # For Deribit
            if self.exchange_id == 'deribit':
                # Get all instruments for the base currency
                instruments = self.exchange.public_get_instruments({
                    'currency': self.base_currency,
                    'kind': 'option',
                    'expired': False
                })

                if not instruments or 'result' not in instruments:
                    logger.warning("No options data available from Deribit")
                    self.options_data = self._simulate_options_data()
                    return True

                # Filter instruments by expiry date
                filtered_instruments = []
                for instr in instruments['result']:
                    expiry_date = datetime.strptime(instr['expiration_timestamp'], '%Y-%m-%d %H:%M:%S')
                    if min_date <= expiry_date <= max_date:
                        filtered_instruments.append(instr)

                # Group by expiry date
                expiry_groups = {}
                for instr in filtered_instruments:
                    expiry = instr['expiration_timestamp'][:10]  # YYYY-MM-DD
                    if expiry not in expiry_groups:
                        expiry_groups[expiry] = []
                    expiry_groups[expiry].append(instr)

                # Get market data for each instrument
                options_data = []

                for expiry, instruments in expiry_groups.items():
                    for instr in instruments:
                        instrument_name = instr['instrument_name']

                        # Get ticker data
                        ticker = self.exchange.public_get_ticker({
                            'instrument_name': instrument_name
                        })

                        if ticker and 'result' in ticker:
                            ticker_data = ticker['result']

                            # Extract relevant data
                            option_type = 'call' if 'C' in instrument_name.split('-')[-1] else 'put'
                            strike = float(instrument_name.split('-')[-2])

                            # Calculate implied volatility from mark price
                            mark_price = ticker_data.get('mark_price', 0)
                            underlying_price = ticker_data.get('underlying_price', 0)

                            if mark_price > 0 and underlying_price > 0:
                                # Time to expiration in years
                                expiry_date = datetime.strptime(instr['expiration_timestamp'], '%Y-%m-%d %H:%M:%S')
                                days_to_expiry = (expiry_date - now).days
                                time_to_expiry = days_to_expiry / 365.0

                                # Calculate implied volatility
                                try:
                                    iv = self._black_scholes_implied_volatility(
                                        option_type, mark_price, underlying_price,
                                        strike, time_to_expiry, self.risk_free_rate
                                    )

                                    # Calculate delta
                                    delta = self._black_scholes_delta(
                                        option_type, underlying_price, strike,
                                        time_to_expiry, self.risk_free_rate, iv
                                    )

                                    # Only include options within the delta range
                                    if delta_range is None or (delta_range[0] <= abs(delta) <= delta_range[1]):
                                        options_data.append({
                                            'instrument': instrument_name,
                                            'type': option_type,
                                            'strike': strike,
                                            'expiry': expiry,
                                            'days_to_expiry': days_to_expiry,
                                            'mark_price': mark_price,
                                            'underlying_price': underlying_price,
                                            'implied_volatility': iv,
                                            'delta': delta
                                        })
                                except:
                                    # Skip if IV calculation fails
                                    pass

                self.options_data = pd.DataFrame(options_data)

                if len(self.options_data) > 0:
                    logger.info(f"Successfully fetched {len(self.options_data)} options")
                    return True
                else:
                    logger.warning("No valid options data available, using simulation")
                    self.options_data = self._simulate_options_data()
                    return True

            # For other exchanges, implement similar logic adapted to their API

            # If exchange not specifically implemented, use simulation
            self.options_data = self._simulate_options_data()
            logger.info("Using simulated options data (exchange API not implemented)")
            return True

        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            # Fall back to simulated data
            self.options_data = self._simulate_options_data()
            return True

    def _simulate_options_data(self):
        """
        Simulate options data when actual options data is not available.

        Returns:
        --------
        pd.DataFrame
            Simulated options data
        """
        # Get the current price of the underlying
        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker['last'] if 'last' in ticker else None

        if current_price is None:
            # If price not available, use a placeholder
            current_price = 50000 if self.base_currency == 'BTC' else 2000 if self.base_currency == 'ETH' else 1000

        # Define expiry dates
        expiry_days = [7, 14, 30, 60, 90]

        # Define strikes (as percentages of current price)
        strike_percentages = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]

        # Define base volatility
        base_volatility = 0.8 if self.base_currency == 'BTC' else 1.0 if self.base_currency == 'ETH' else 1.2

        # Generate simulated data
        options_data = []

        for days in expiry_days:
            expiry_date = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
            time_to_expiry = days / 365.0

            # Volatility tends to be higher for longer expirations (term structure)
            time_factor = 1.0 + 0.1 * np.log(1 + days / 30.0)

            for pct in strike_percentages:
                strike = current_price * pct

                # Volatility smile (higher for out-of-the-money options)
                moneyness = abs(np.log(strike / current_price))
                smile_factor = 1.0 + 0.2 * moneyness ** 2

                # Calculate simulated implied volatility
                iv = base_volatility * time_factor * smile_factor

                # Add some random noise
                iv *= np.random.normal(1.0, 0.05)

                # Limit IV to reasonable range
                iv = max(0.1, min(iv, 2.0))

                # Add call option
                delta_call = self._black_scholes_delta(
                    'call', current_price, strike, time_to_expiry, self.risk_free_rate, iv
                )

                options_data.append({
                    'instrument': f"{self.base_currency}-{expiry_date}-{strike}-C",
                    'type': 'call',
                    'strike': strike,
                    'expiry': expiry_date,
                    'days_to_expiry': days,
                    'mark_price': None,  # Not needed for simulation
                    'underlying_price': current_price,
                    'implied_volatility': iv,
                    'delta': delta_call
                })

                # Add put option
                delta_put = self._black_scholes_delta(
                    'put', current_price, strike, time_to_expiry, self.risk_free_rate, iv
                )

                options_data.append({
                    'instrument': f"{self.base_currency}-{expiry_date}-{strike}-P",
                    'type': 'put',
                    'strike': strike,
                    'expiry': expiry_date,
                    'days_to_expiry': days,
                    'mark_price': None,  # Not needed for simulation
                    'underlying_price': current_price,
                    'implied_volatility': iv,
                    'delta': delta_put
                })

        return pd.DataFrame(options_data)

    def fetch_futures_data(self):
        """
        Fetch futures data from the exchange.

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching futures data for {self.symbol}...")

            # Get all futures contracts for the symbol
            markets = self.exchange.load_markets()
            futures = []

            for market_id, market in markets.items():
                if market['type'] == 'future' and market['base'] == self.base_currency:
                    futures.append(market)

            if not futures:
                logger.warning("No futures data available")
                return False

            # Fetch ticker data for each future
            futures_data = []

            for future in futures:
                symbol = future['symbol']

                # Get ticker
                ticker = self.exchange.fetch_ticker(symbol)

                if ticker:
                    # Calculate days to expiry
                    if 'expiry' in future and future['expiry']:
                        expiry_timestamp = future['expiry'] / 1000.0 if isinstance(future['expiry'], int) else 0
                        if expiry_timestamp > 0:
                            expiry_date = datetime.fromtimestamp(expiry_timestamp)
                            days_to_expiry = (expiry_date - datetime.now()).days
                        else:
                            days_to_expiry = None  # Perpetual futures
                    else:
                        days_to_expiry = None

                    # Get spot price
                    spot_ticker = self.exchange.fetch_ticker(self.symbol)
                    spot_price = spot_ticker['last'] if 'last' in spot_ticker else None

                    if spot_price and ticker.get('last'):
                        # Calculate futures premium/discount
                        premium = (ticker['last'] / spot_price - 1) * 100  # As percentage

                        futures_data.append({
                            'symbol': symbol,
                            'last_price': ticker['last'],
                            'volume': ticker.get('volume', 0),
                            'spot_price': spot_price,
                            'premium_pct': premium,
                            'days_to_expiry': days_to_expiry,
                            'expiry': expiry_date.strftime('%Y-%m-%d') if days_to_expiry is not None else 'perpetual'
                        })

            self.futures_data = pd.DataFrame(futures_data)

            if len(self.futures_data) > 0:
                logger.info(f"Successfully fetched {len(self.futures_data)} futures contracts")
                return True
            else:
                logger.warning("No valid futures data available")
                return False

        except Exception as e:
            logger.error(f"Error fetching futures data: {e}")
            return False

    def fetch_funding_rates(self):
        """
        Fetch funding rates for perpetual contracts.

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching funding rates for {self.base_currency}...")

            # Check if the exchange provides funding rates
            if not hasattr(self.exchange, 'fetch_funding_rates'):
                logger.warning("Exchange does not provide funding rates API")
                return False

            # Fetch funding rates
            funding_rates = self.exchange.fetch_funding_rates([self.symbol])

            if not funding_rates or self.symbol not in funding_rates:
                logger.warning("No funding rates available")
                return False

            rate_data = funding_rates[self.symbol]

            # Add timestamp
            rate_data['timestamp'] = datetime.now().isoformat()

            # Convert to DataFrame
            self.funding_rates = pd.DataFrame([rate_data])

            logger.info(f"Successfully fetched funding rates: {rate_data}")
            return True

        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return False

    def fetch_historical_funding_rates(self, days=30):
        """
        Fetch historical funding rates for analysis.

        Parameters:
        -----------
        days : int
            Number of days of historical data to fetch

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching historical funding rates for {self.base_currency}...")

            # Check if the exchange provides historical funding rates
            if not hasattr(self.exchange, 'fetch_funding_rate_history'):
                logger.warning("Exchange does not provide historical funding rates API")
                return False

            # Calculate start time
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Fetch historical funding rates
            funding_history = self.exchange.fetch_funding_rate_history(self.symbol, since)

            if not funding_history:
                logger.warning("No historical funding rates available")
                return False

            # Convert to DataFrame
            history_df = pd.DataFrame(funding_history)

            # Convert timestamp to datetime
            if 'timestamp' in history_df.columns:
                history_df['datetime'] = pd.to_datetime(history_df['timestamp'], unit='ms')

            self.historical_funding_rates = history_df

            logger.info(f"Successfully fetched {len(history_df)} historical funding rates")
            return True

        except Exception as e:
            logger.error(f"Error fetching historical funding rates: {e}")
            return False

    def calculate_volatility_surface(self):
        """
        Calculate implied volatility surface from options data.

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if self.options_data is None:
                logger.warning("No options data available. Fetch options data first.")
                return False

            if len(self.options_data) == 0:
                logger.warning("Empty options data.")
                return False

            logger.info("Calculating volatility surface...")

            # Group options by expiry date
            expiry_groups = self.options_data.groupby('days_to_expiry')

            # Create volatility surface data
            surface_data = []

            for days_to_expiry, group in expiry_groups:
                if len(group) < 3:
                    continue  # Need at least 3 points for a meaningful curve

                # Get the spot price
                spot_price = group['underlying_price'].iloc[0]

                # Calculate moneyness (log of strike / spot)
                group = group.copy()
                group['moneyness'] = np.log(group['strike'] / spot_price)

                # Group by call/put
                for option_type, type_group in group.groupby('type'):
                    # Sort by moneyness
                    type_group = type_group.sort_values('moneyness')

                    # Get moneyness and IV arrays
                    moneyness = type_group['moneyness'].values
                    iv = type_group['implied_volatility'].values

                    # Create a smooth IV curve using cubic spline
                    try:
                        if len(moneyness) >= 3:  # Need at least 3 points for cubic spline
                            cs = CubicSpline(moneyness, iv)

                            # Generate points along the curve
                            x_new = np.linspace(min(moneyness), max(moneyness), 50)
                            y_new = cs(x_new)

                            # Add to surface data
                            for m, v in zip(x_new, y_new):
                                surface_data.append({
                                    'days_to_expiry': days_to_expiry,
                                    'moneyness': m,
                                    'implied_volatility': v,
                                    'option_type': option_type
                                })
                    except Exception as e:
                        logger.warning(f"Error in cubic spline calculation: {e}")

            if len(surface_data) == 0:
                logger.warning("Could not calculate volatility surface.")
                return False

            self.volatility_surface = pd.DataFrame(surface_data)

            logger.info("Volatility surface calculated successfully.")
            return True

        except Exception as e:
            logger.error(f"Error calculating volatility surface: {e}")
            return False

    def calculate_volatility_term_structure(self):
        """
        Calculate implied volatility term structure from options data.

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if self.options_data is None:
                logger.warning("No options data available. Fetch options data first.")
                return False

            if len(self.options_data) == 0:
                logger.warning("Empty options data.")
                return False

            logger.info("Calculating volatility term structure...")

            # Calculate at-the-money (ATM) IV for each expiry
            term_structure_data = []

            for days, group in self.options_data.groupby('days_to_expiry'):
                if len(group) < 2:
                    continue

                # Get spot price
                spot_price = group['underlying_price'].iloc[0]

                # Calculate moneyness
                group = group.copy()
                group['moneyness'] = abs(np.log(group['strike'] / spot_price))

                # Find near-ATM options (smallest absolute moneyness)
                near_atm = group.nsmallest(3, 'moneyness')

                if len(near_atm) > 0:
                    # Calculate average IV of near-ATM options
                    atm_iv = near_atm['implied_volatility'].mean()

                    term_structure_data.append({
                        'days_to_expiry': days,
                        'atm_implied_volatility': atm_iv
                    })

            if len(term_structure_data) == 0:
                logger.warning("Could not calculate volatility term structure.")
                return False

            # Create DataFrame and sort by days to expiry
            self.volatility_term_structure = pd.DataFrame(term_structure_data).sort_values('days_to_expiry')

            logger.info("Volatility term structure calculated successfully.")
            return True

        except Exception as e:
            logger.error(f"Error calculating volatility term structure: {e}")
            return False

    def calculate_implied_volatility_index(self):
        """
        Calculate a VIX-like implied volatility index for the cryptocurrency.

        Returns:
        --------
        float
            Implied volatility index value
        """
        try:
            # If we have options data, calculate IV index
            if self.options_data is not None and len(self.options_data) > 0:
                # Calculate VIX-like index using near-term options
                # (simplified implementation)

                # Filter options with 23-37 days to expiry (close to 30 days)
                near_term = self.options_data[
                    (self.options_data['days_to_expiry'] >= 23) &
                    (self.options_data['days_to_expiry'] <= 37)
                    ]

                if len(near_term) == 0:
                    # If no options in ideal range, use closest available
                    median_days = self.options_data['days_to_expiry'].median()
                    near_term = self.options_data[
                        (self.options_data['days_to_expiry'] >= median_days * 0.8) &
                        (self.options_data['days_to_expiry'] <= median_days * 1.2)
                        ]

                if len(near_term) > 0:
                    # Calculate average IV weighted by delta (ATM options get higher weight)
                    near_term['delta_weight'] = near_term['delta'].apply(lambda x: 1 - abs(abs(x) - 0.5))
                    iv_index = np.average(
                        near_term['implied_volatility'],
                        weights=near_term['delta_weight']
                    )

                    return iv_index * 100  # Convert to percentage points like VIX

            # If no options data or calculation failed, use alternatives

            # Try using funding rates if available
            if self.funding_rates is not None and len(self.funding_rates) > 0:
                # Higher absolute funding rates often correlate with higher expected volatility
                funding_rate = abs(self.funding_rates['rate'].iloc[0])

                # Convert funding rate to annualized volatility
                # This is a heuristic approximation
                iv_proxy = funding_rate * 100 * 10  # Scale factor is empirical

                return min(iv_proxy, 150)  # Cap at reasonable maximum

            # If futures data is available, use futures premium
            if self.futures_data is not None and len(self.futures_data) > 0:
                # Calculate average premium across futures
                avg_premium = abs(self.futures_data['premium_pct'].mean())

                # Convert premium to volatility estimate
                # This is another heuristic approximation
                iv_proxy = avg_premium * 0.5  # Scale factor is empirical

                return min(iv_proxy, 150)  # Cap at reasonable maximum

            # If all else fails, return a default value based on historical data
            # (Should be calibrated to the specific cryptocurrency)
            default_iv = 80 if self.base_currency == 'BTC' else 100 if self.base_currency == 'ETH' else 120

            logger.warning("Using default IV value due to insufficient data")
            return default_iv

        except Exception as e:
            logger.error(f"Error calculating implied volatility index: {e}")
            return 80  # Default fallback

    def analyze_volatility_skew(self):
        """
        Analyze the volatility skew (smile) in the options data.

        Returns:
        --------
        dict
            Skew analysis results
        """
        try:
            if self.options_data is None or len(self.options_data) == 0:
                return {
                    'error': 'No options data available',
                    'skew': None,
                    'interpretation': None
                }

            # Calculate average days to expiry for analysis
            avg_days = self.options_data['days_to_expiry'].median()

            # Filter options around the average expiry
            filtered_options = self.options_data[
                (self.options_data['days_to_expiry'] >= avg_days * 0.8) &
                (self.options_data['days_to_expiry'] <= avg_days * 1.2)
                ]

            if len(filtered_options) < 5:
                # If too few options, use all available
                filtered_options = self.options_data

            # Calculate moneyness
            filtered_options = filtered_options.copy()
            filtered_options['moneyness'] = np.log(filtered_options['strike'] / filtered_options['underlying_price'])

            # Calculate skew (difference between OTM put and call IV)
            otm_puts = filtered_options[
                (filtered_options['type'] == 'put') &
                (filtered_options['moneyness'] < -0.05)
                ]

            otm_calls = filtered_options[
                (filtered_options['type'] == 'call') &
                (filtered_options['moneyness'] > 0.05)
                ]

            if len(otm_puts) > 0 and len(otm_calls) > 0:
                avg_otm_put_iv = otm_puts['implied_volatility'].mean()
                avg_otm_call_iv = otm_calls['implied_volatility'].mean()

                skew = avg_otm_put_iv - avg_otm_call_iv

                # Interpret the skew
                if skew > 0.1:
                    interpretation = "strong_negative_skew"
                    message = "Market expects downside risk, potential for sharp price drops"
                elif skew > 0.05:
                    interpretation = "moderate_negative_skew"
                    message = "Some concern about downside risk"
                elif skew < -0.1:
                    interpretation = "strong_positive_skew"
                    message = "Market anticipates upside volatility, potential for sharp price increases"
                elif skew < -0.05:
                    interpretation = "moderate_positive_skew"
                    message = "Some expectation of upside volatility"
                else:
                    interpretation = "neutral"
                    message = "Balanced market expectations"

                return {
                    'skew': skew,
                    'otm_put_iv': avg_otm_put_iv,
                    'otm_call_iv': avg_otm_call_iv,
                    'interpretation': interpretation,
                    'message': message
                }

            # If not enough data for skew calculation, return None
            return {
                'error': 'Insufficient data for skew calculation',
                'skew': None,
                'interpretation': None
            }

        except Exception as e:
            logger.error(f"Error analyzing volatility skew: {e}")
            return {
                'error': str(e),
                'skew': None,
                'interpretation': None
            }

    def calculate_put_call_ratio(self):
        """
        Calculate put-call ratio from options data.

        Returns:
        --------
        dict
            Put-call ratio analysis
        """
        try:
            if self.options_data is None or len(self.options_data) == 0:
                return {
                    'error': 'No options data available',
                    'put_call_ratio': None,
                    'interpretation': None
                }

            # Count puts and calls
            puts = self.options_data[self.options_data['type'] == 'put']
            calls = self.options_data[self.options_data['type'] == 'call']

            if len(calls) == 0:
                return {
                    'error': 'No call options in the dataset',
                    'put_call_ratio': None,
                    'interpretation': None
                }

            # Calculate put-call ratio
            put_call_ratio = len(puts) / len(calls)

            # Interpret the ratio
            if put_call_ratio > 1.5:
                interpretation = "very_bearish"
                message = "High demand for puts indicates strong bearish sentiment"
            elif put_call_ratio > 1.2:
                interpretation = "bearish"
                message = "Elevated put buying suggests bearish outlook"
            elif put_call_ratio < 0.5:
                interpretation = "very_bullish"
                message = "Low put demand indicates strong bullish sentiment"
            elif put_call_ratio < 0.8:
                interpretation = "bullish"
                message = "Reduced put buying suggests bullish outlook"
            else:
                interpretation = "neutral"
                message = "Balanced options activity"

            return {
                'put_call_ratio': put_call_ratio,
                'put_count': len(puts),
                'call_count': len(calls),
                'interpretation': interpretation,
                'message': message
            }

        except Exception as e:
            logger.error(f"Error calculating put-call ratio: {e}")
            return {
                'error': str(e),
                'put_call_ratio': None,
                'interpretation': None
            }

    def analyze_term_structure_slope(self):
        """
        Analyze the slope of the volatility term structure.

        Returns:
        --------
        dict
            Term structure analysis
        """
        try:
            if self.volatility_term_structure is None or len(self.volatility_term_structure) < 2:
                return {
                    'error': 'Insufficient term structure data',
                    'slope': None,
                    'interpretation': None
                }

            # Sort by days to expiry
            term_structure = self.volatility_term_structure.sort_values('days_to_expiry')

            # Calculate log of days to expiry
            term_structure['log_days'] = np.log(term_structure['days_to_expiry'])

            # Perform linear regression
            x = term_structure['log_days'].values
            y = term_structure['atm_implied_volatility'].values

            # Calculate slope using linear regression
            slope, intercept, r_value, p_value, std_err = np.polyfit(x, y, 1, full=True)[0:5]

            # Convert to percent slope per doubling of time
            slope_per_doubling = slope * np.log(2)

            # Calculate term structure curve
            term_structure['fitted_iv'] = intercept + slope * term_structure['log_days']

            # Calculate residuals
            term_structure['residual'] = term_structure['atm_implied_volatility'] - term_structure['fitted_iv']

            # Calculate average absolute residual
            avg_abs_residual = term_structure['residual'].abs().mean()

            # Interpret the slope
            if slope_per_doubling > 0.10:
                interpretation = "steep_contango"
                message = "Strong anticipation of increasing volatility in the future"
            elif slope_per_doubling > 0.05:
                interpretation = "moderate_contango"
                message = "Expectation of slightly higher future volatility"
            elif slope_per_doubling < -0.10:
                interpretation = "steep_backwardation"
                message = "Strong expectation of decreasing volatility"
            elif slope_per_doubling < -0.05:
                interpretation = "moderate_backwardation"
                message = "Expectation of slightly lower future volatility"
            else:
                interpretation = "flat"
                message = "No significant expected change in volatility across time"
            return {
                'slope': slope,
                'slope_per_doubling': slope_per_doubling,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'avg_abs_residual': avg_abs_residual,
                'interpretation': interpretation,
                'message': message,
                'term_structure': term_structure.to_dict(orient='records')
            }

        except Exception as e:
            logger.error(f"Error analyzing term structure slope: {e}")
            return {
                'error': str(e),
                'slope': None,
                'interpretation': None
            }

    def _black_scholes_implied_volatility(self, option_type, price, spot, strike, time_to_expiry, risk_free_rate):
        """
        Calculate implied volatility using the Black-Scholes model.

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        price : float
            Option price
        spot : float
            Spot price of the underlying
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiry in years
        risk_free_rate : float
            Risk-free interest rate

        Returns:
        --------
        float
            Implied volatility
        """
        from scipy.optimize import brentq

        def _black_scholes_price(sigma):
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (
                    sigma * np.sqrt(time_to_expiry))
            d2 = d1 - sigma * np.sqrt(time_to_expiry)

            if option_type.lower() == 'call':
                price = spot * self._norm_cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * self._norm_cdf(
                    d2)
            else:  # put
                price = strike * np.exp(-risk_free_rate * time_to_expiry) * self._norm_cdf(-d2) - spot * self._norm_cdf(
                    -d1)

            return price

        def _objective(sigma):
            return _black_scholes_price(sigma) - price

        try:
            # Reasonable range for implied volatility
            return brentq(_objective, 0.001, 5.0)
        except:
            # If the solver fails, try a broader range
            try:
                return brentq(_objective, 0.0001, 10.0)
            except:
                # If it still fails, return a default value
                return 1.0

    def _black_scholes_delta(self, option_type, spot, strike, time_to_expiry, risk_free_rate, volatility):
        """
        Calculate the delta of an option using the Black-Scholes model.

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        spot : float
            Spot price of the underlying
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiry in years
        risk_free_rate : float
            Risk-free interest rate
        volatility : float
            Implied volatility

        Returns:
        --------
        float
            Option delta
        """
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry))

        if option_type.lower() == 'call':
            return self._norm_cdf(d1)
        else:  # put
            return self._norm_cdf(d1) - 1.0

    def _norm_cdf(self, x):
        """
        Standard normal cumulative distribution function.
        """
        return (1.0 + np.erf(x / np.sqrt(2.0))) / 2.0

    def visualize_volatility_surface(self, show_plot=True):
        """
        Visualize the implied volatility surface.

        Parameters:
        -----------
        show_plot : bool
            Whether to display the plot or return the figure

        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if show_plot is False, None otherwise
        """
        if self.volatility_surface is None or len(self.volatility_surface) == 0:
            logger.warning("No volatility surface data available.")
            return None

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Create figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Extract data for plotting
        x = self.volatility_surface['moneyness'].values
        y = self.volatility_surface['days_to_expiry'].values
        z = self.volatility_surface['implied_volatility'].values
        colors = np.zeros_like(z)

        # Color by option type
        for i, option_type in enumerate(self.volatility_surface['option_type']):
            colors[i] = 0 if option_type == 'call' else 1

        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=30, alpha=0.7)

        # Add labels and title
        ax.set_xlabel('Moneyness (log(K/S))')
        ax.set_ylabel('Days to Expiry')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'Implied Volatility Surface for {self.symbol}')

        # Add colorbar for option type
        cbar = plt.colorbar(scatter)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Call', 'Put'])

        # Show plot or return figure
        plt.tight_layout()

        if show_plot:
            plt.show()
            return None
        else:
            return fig

    def visualize_term_structure(self, show_plot=True):
        """
        Visualize the volatility term structure.

        Parameters:
        -----------
        show_plot : bool
            Whether to display the plot or return the figure

        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if show_plot is False, None otherwise
        """
        if self.volatility_term_structure is None or len(self.volatility_term_structure) == 0:
            logger.warning("No term structure data available.")
            return None

        import matplotlib.pyplot as plt

        # Sort by days to expiry
        term_structure = self.volatility_term_structure.sort_values('days_to_expiry')

        # Create figure
        fig = plt.figure(figsize=(12, 8))

        # Plot term structure
        plt.plot(term_structure['days_to_expiry'], term_structure['atm_implied_volatility'],
                 'o-', lw=2, markersize=8)

        # Add labels and title
        plt.xlabel('Days to Expiry')
        plt.ylabel('ATM Implied Volatility')
        plt.title(f'Volatility Term Structure for {self.symbol}')
        plt.grid(True, alpha=0.3)

        # Show plot or return figure
        plt.tight_layout()

        if show_plot:
            plt.show()
            return None
        else:
            return fig

    def visualize_volatility_skew(self, show_plot=True):
        """
        Visualize the volatility skew (smile).

        Parameters:
        -----------
        show_plot : bool
            Whether to display the plot or return the figure

        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if show_plot is False, None otherwise
        """
        if self.options_data is None or len(self.options_data) == 0:
            logger.warning("No options data available.")
            return None

        import matplotlib.pyplot as plt

        # Calculate average days to expiry for analysis
        avg_days = self.options_data['days_to_expiry'].median()

        # Filter options around the average expiry
        filtered_options = self.options_data[
            (self.options_data['days_to_expiry'] >= avg_days * 0.8) &
            (self.options_data['days_to_expiry'] <= avg_days * 1.2)
            ]

        if len(filtered_options) < 5:
            # If too few options, use all available
            filtered_options = self.options_data

        # Calculate moneyness
        filtered_options = filtered_options.copy()
        filtered_options['moneyness'] = np.log(filtered_options['strike'] / filtered_options['underlying_price'])

        # Create figure
        fig = plt.figure(figsize=(12, 8))

        # Plot calls and puts separately
        for option_type, color, marker in [('call', 'blue', 'o'), ('put', 'red', '^')]:
            options = filtered_options[filtered_options['type'] == option_type]

            if len(options) > 0:
                plt.scatter(options['moneyness'], options['implied_volatility'],
                            color=color, marker=marker, s=80, alpha=0.7,
                            label=f'{option_type.capitalize()} Options')

                # Add trendline
                if len(options) >= 2:
                    try:
                        z = np.polyfit(options['moneyness'], options['implied_volatility'], 2)
                        p = np.poly1d(z)

                        x_range = np.linspace(min(options['moneyness']), max(options['moneyness']), 100)
                        plt.plot(x_range, p(x_range), '-', color=color, alpha=0.5)
                    except:
                        pass

        # Add vertical line at ATM
        plt.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='At-the-Money')

        # Add labels and title
        plt.xlabel('Moneyness (log(K/S))')
        plt.ylabel('Implied Volatility')
        plt.title(f'Volatility Skew for {self.symbol} (Expiry around {avg_days} days)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Show plot or return figure
        plt.tight_layout()

        if show_plot:
            plt.show()
            return None
        else:
            return fig

    def run_analysis(self):
        """
        Run a complete implied volatility analysis.

        Returns:
        --------
        dict
            Analysis results
        """
        logger.info(f"Running implied volatility analysis for {self.symbol}...")

        # Fetch data
        self.fetch_options_data()
        self.fetch_futures_data()
        self.fetch_funding_rates()

        # Calculate volatility metrics
        if self.options_data is not None and len(self.options_data) > 0:
            self.calculate_volatility_surface()
            self.calculate_volatility_term_structure()

        # Calculate IV index
        iv_index = self.calculate_implied_volatility_index()

        # Analyze skew
        skew_analysis = self.analyze_volatility_skew()

        # Calculate put-call ratio
        put_call_ratio = self.calculate_put_call_ratio()

        # Analyze term structure
        term_structure = self.analyze_term_structure_slope()

        # Create a comprehensive report
        report = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'implied_volatility_index': iv_index,
            'volatility_skew': skew_analysis.get('skew'),
            'skew_interpretation': skew_analysis.get('interpretation'),
            'skew_message': skew_analysis.get('message'),
            'put_call_ratio': put_call_ratio.get('put_call_ratio'),
            'put_call_interpretation': put_call_ratio.get('interpretation'),
            'term_structure_slope': term_structure.get('slope_per_doubling'),
            'term_structure_interpretation': term_structure.get('interpretation'),
            'funding_rate': self.funding_rates['rate'].iloc[0] if self.funding_rates is not None and len(
                self.funding_rates) > 0 else None,
            'futures_premium': self.futures_data['premium_pct'].mean() if self.futures_data is not None and len(
                self.futures_data) > 0 else None,
            'data_quality': 'high' if self.options_data is not None and len(
                self.options_data) > 20 else 'medium' if self.options_data is not None and len(
                self.options_data) > 0 else 'low'
        }

        logger.info("Implied volatility analysis completed.")
        return report

    # Factory function to get an implied volatility analyzer
    def get_implied_volatility_analyzer(exchange_id='deribit', symbol='BTC/USD', api_key=None, api_secret=None):
        """
        Get a configured implied volatility analyzer.

        Parameters:
        -----------
        exchange_id : str
            The ID of the exchange to use
        symbol : str
            The trading pair to analyze
        api_key : str, optional
            API key for accessing exchange API
        api_secret : str, optional
            API secret for accessing exchange API

        Returns:
        --------
        ImpliedVolatilityAnalyzer
            Configured analyzer instance
        """
        return ImpliedVolatilityAnalyzer(
            exchange_id=exchange_id,
            symbol=symbol,
            api_key=api_key,
            api_secret=api_secret
        )