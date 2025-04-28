"""
Progressive Adaptive Volatility Indicator for cryptocurrency markets.
Main composite indicator that combines multiple volatility analysis techniques.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import logging
import json
from functools import lru_cache
import ccxt  # For exchange data
from ccxt.base.errors import NetworkError, ExchangeError
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# Измените импорты на абсолютные
from crypto_volatility_indicator.utils.logger import get_logger
from crypto_volatility_indicator.utils.helpers import (
    calculate_returns, calculate_volatility, calculate_rolling_volatility,
    calculate_z_score, calculate_hurst_exponent, calculate_fractal_dimension,
    timeframe_to_seconds, save_dataframe, load_dataframe, cached
)
from crypto_volatility_indicator.engine.volatility.micro import MicroVolatilityAnalyzer
from crypto_volatility_indicator.engine.volatility.meso import MesoVolatilityAnalyzer
from crypto_volatility_indicator.engine.volatility.macro import MacroVolatilityAnalyzer
from crypto_volatility_indicator.engine.analysis.regime_detector import RegimeDetector
from crypto_volatility_indicator.engine.analysis.fractal_analyzer import FractalAnalyzer
from crypto_volatility_indicator.engine.analysis.fractal_analyzer import get_fractal_analyzer
from crypto_volatility_indicator.engine.analysis.cycle_analyzer import CycleAnalyzer
from crypto_volatility_indicator.engine.analysis.implied_vol import ImpliedVolatilityAnalyzer
from crypto_volatility_indicator.engine.models.hybrid_model import VolatilityPredictionModel
from crypto_volatility_indicator.engine.indicators.kama import KAMAIndicator
from crypto_volatility_indicator.engine.indicators.signals import SignalGenerator

logger = get_logger(__name__)


from crypto_volatility_indicator.engine.indicators.kama import KAMAIndicator

class ProgressiveAdaptiveVolatilityIndicator:
    """
    Progressive Adaptive Volatility Indicator for cryptocurrency markets.

    This indicator combines multiple analysis techniques:
    - Multi-timeframe volatility analysis (micro, meso, macro)
    - Adaptive KAMA with dynamic parameters
    - Market regime detection and adaptation
    - Fractal analysis of volatility
    - Cycle analysis
    - Implied volatility integration (from options markets)
    - Machine learning predictions
    """

    def __init__(self, config=None):
        """
        Initialize the indicator with configuration.

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        self.config = config or {}

        # Load settings from config
        self.assets = self.config.get('assets', ['BTC/USDT'])
        self.exchange_id = self.config.get('exchange_id', 'binance')
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        self.history_limit = self.config.get('history_limit', 1000)  # Number of candlesticks to fetch
        self.update_interval = self.config.get('update_interval', 60)  # Seconds

        # Set up data storage
        self.data_dir = self.config.get('data_dir', 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize exchange
        self.exchange = self._initialize_exchange()

        # Initialize component analyzers
        self.micro_analyzer = MicroVolatilityAnalyzer(config.get('micro_volatility', {}))
        self.meso_analyzer = MesoVolatilityAnalyzer(config.get('meso_volatility', {}))
        self.macro_analyzer = MacroVolatilityAnalyzer(config.get('macro_volatility', {}))
        regime_config = config.get('regime_detector', {})
        n_regimes = regime_config.get('n_regimes', 4)
        window_size = regime_config.get('window_size', 20)
        self.regime_detector = RegimeDetector(n_regimes=n_regimes, window_size=window_size)
        self.fractal_analyzer = get_fractal_analyzer(config.get('fractal_analyzer', {}))
        self.cycle_analyzer = CycleAnalyzer(config.get('cycle_analyzer', {}))
        self.implied_vol_analyzer = ImpliedVolatilityAnalyzer(
            exchange_id=self.config.get('exchange_id', 'deribit'),
            symbol=self.config.get('implied_vol_symbol', 'BTC/USD')
        )

        # Initialize adaptive KAMA
        self.kama_params = self.config.get('kama', {})
        self.kama = {}  # One KAMA per asset

        # Initialize prediction model
        self.use_ml = self.config.get('use_ml', True)
        self.prediction_model = None
        if self.use_ml:
            self.prediction_model = VolatilityPredictionModel(config.get('prediction_model', {}))

        # Initialize signal generator
        self.signal_generator = SignalGenerator(config.get('signals', {}))

        # Data caches
        self.price_data = {}  # Asset -> DataFrame
        self.volatility_data = {}  # Asset -> DataFrame
        self.regime_data = {}  # Asset -> DataFrame
        self.predictions = {}  # Asset -> Series
        self.signals = {}  # Asset -> DataFrame

        # Background update thread
        self.update_thread_active = False
        self.update_thread = None

        # Initialization status
        self.initialized = False

    def _initialize_exchange(self):
        """
        Initialize exchange connection for data fetching.

        Returns:
        --------
        ccxt.Exchange
            Initialized exchange object
        """
        try:
            # Get exchange class
            exchange_class = getattr(ccxt, self.exchange_id)

            # Initialize with config options
            exchange_config = self.config.get('exchange_config', {})
            exchange = exchange_class(exchange_config)

            # Set rate limits
            exchange.enableRateLimit = True

            logger.info(f"Exchange {self.exchange_id} initialized")
            return exchange

        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            raise

    def initialize_assets(self):
        """Initialize data for all configured assets."""
        for asset in self.assets:
            try:
                logger.info(f"Initializing data for {asset}")

                # Fetch initial price data
                self._initialize_price_data(asset)

                # Calculate initial volatility data
                self._initialize_volatility_data(asset)

                # Initialize KAMA
                self._initialize_kama(asset)

                # Initialize regime data
                self._initialize_regime_data(asset)

                # Initialize prediction model
                if self.use_ml and self.prediction_model:
                    self._initialize_predictions(asset)

                # Initialize signals
                self._initialize_signals(asset)

                logger.info(f"Initialization complete for {asset}")

            except Exception as e:
                logger.error(f"Error initializing {asset}: {str(e)}")
                raise

        # Set initialization flag
        self.initialized = True

        # Connect signal generator to this indicator
        try:
            self.signal_generator.set_volatility_indicator(self)
            logger.info("Signal generator connected to volatility indicator")
        except Exception as e:
            logger.error(f"Error connecting signal generator: {e}")

    def initialize_with_test_data(self, start_date, end_date):
        """
        Initialize using test data for backtesting.

        Parameters:
        -----------
        start_date : datetime
            Start date for test data
        end_date : datetime
            End date for test data
        """
        logger.info(f"Initializing with test data from {start_date} to {end_date}")

        for asset in self.assets:
            try:
                # Load data from file if available
                data_file = os.path.join(self.data_dir, f"{asset.replace('/', '_')}_test.csv")

                if os.path.exists(data_file):
                    logger.info(f"Loading test data for {asset} from {data_file}")
                    price_data = pd.read_csv(data_file, index_col=0, parse_dates=True)

                    # Filter by date range
                    price_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
                else:
                    # Generate synthetic test data
                    logger.info(f"Generating synthetic test data for {asset}")
                    price_data = self._generate_test_data(asset, start_date, end_date)

                    # Save for future use
                    price_data.to_csv(data_file)

                # Store price data
                self.price_data[asset] = price_data

                # Calculate volatility data
                self._initialize_volatility_data(asset)

                # Initialize KAMA
                self._initialize_kama(asset)

                # Initialize regime data
                self._initialize_regime_data(asset)

                # Initialize prediction model
                if self.use_ml and self.prediction_model:
                    self._initialize_predictions(asset)

                # Initialize signals
                self._initialize_signals(asset)

                logger.info(f"Test initialization complete for {asset}")

            except Exception as e:
                logger.error(f"Error initializing test data for {asset}: {str(e)}")
                raise

        # Set initialization flag
        self.initialized = True

        # Connect signal generator to this indicator
        self.signal_generator.set_volatility_indicator(self)

        logger.info("All test assets initialized")

        return True

    def _generate_test_data(self, asset, start_date, end_date):
        """
        Generate synthetic test data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        start_date : datetime
            Start date
        end_date : datetime
            End date

        Returns:
        --------
        pd.DataFrame
            Synthetic price data
        """
        # Calculate number of days
        days = (end_date - start_date).days + 1

        # Create index
        index = pd.date_range(start=start_date, end=end_date, freq='1h')

        # Generate random price data with trends and volatility clusters
        np.random.seed(42)  # For reproducibility

        # Base price
        if 'BTC' in asset:
            base_price = 40000
        elif 'ETH' in asset:
            base_price = 2000
        else:
            base_price = 100

        # Generate log returns with autocorrelation and volatility clustering
        n = len(index)
        volatility = np.zeros(n)
        returns = np.zeros(n)

        # Initial volatility
        volatility[0] = 0.02

        # Generate volatility process (GARCH-like)
        for i in range(1, n):
            # Volatility clustering
            volatility[i] = 0.01 + 0.85 * volatility[i - 1] + 0.1 * np.random.normal(0, 0.02) ** 2

            # Returns with volatility
            returns[i] = np.random.normal(0, volatility[i])

            # Add some autocorrelation
            if i > 1:
                returns[i] += 0.05 * returns[i - 1]

        # Add some trends and cycles
        trend = np.linspace(0, 0.5, n) * np.sin(np.linspace(0, 10, n))
        returns += trend

        # Add some seasonality
        day_cycle = np.sin(np.linspace(0, 2 * np.pi * days, n))
        week_cycle = np.sin(np.linspace(0, 2 * np.pi * days / 7, n))

        returns += 0.001 * day_cycle + 0.003 * week_cycle

        # Calculate cumulative returns and prices
        cum_returns = np.cumsum(returns)
        prices = base_price * np.exp(cum_returns)

        # Create DataFrame
        df = pd.DataFrame(index=index, columns=['open', 'high', 'low', 'close', 'volume'])

        # Fill with values
        df['close'] = prices

        # Generate OHLC from close prices
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = df['close'].iloc[0] * (1 - np.random.normal(0, 0.01))

        for i in range(len(df)):
            volatility_factor = max(0.005, volatility[i])
            df.iloc[i, df.columns.get_loc('high')] = df.iloc[i, df.columns.get_loc('close')] * (
                        1 + np.random.uniform(0.001, volatility_factor))
            df.iloc[i, df.columns.get_loc('low')] = df.iloc[i, df.columns.get_loc('close')] * (
                        1 - np.random.uniform(0.001, volatility_factor))

        # Generate volume
        volume_base = np.where(prices < base_price, base_price / prices, prices / base_price)
        volume = volume_base * (1 + 3 * volatility) * np.random.lognormal(0, 0.5, n)
        df['volume'] = volume * base_price * 10

        return df

    def _initialize_price_data(self, asset):
        """
        Initialize price data for an asset by fetching historical data.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        all_data = []

        # Устанавливаем максимальное количество попыток
        max_retries = 3
        retry_delay = 5  # секунды

        # Проверяем доступность биржи
        if self.exchange is None:
            raise ValueError(f"Exchange not initialized properly for {asset}")

        # Fetch data for each timeframe
        for timeframe in self.timeframes:
            for attempt in range(1, max_retries + 1):
                try:
                    # Validate timeframe
                    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d',
                                        '1w', '1M']
                    if timeframe not in valid_timeframes:
                        logger.warning(f"Invalid timeframe: {timeframe}. Using default '1h'.")
                        timeframe = '1h'

                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=asset,
                        timeframe=timeframe,
                        limit=self.history_limit
                    )

                    if not ohlcv:
                        logger.warning(f"No data returned for {asset} ({timeframe})")
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )

                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    # Store the data
                    all_data.append((timeframe, df))

                    logger.debug(f"Fetched {len(df)} candlesticks for {asset} ({timeframe})")

                    # Avoid rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)

                    # Успешно получили данные, выходим из цикла попыток
                    break

                except ccxt.NetworkError as e:
                    if attempt < max_retries:
                        logger.warning(
                            f"Network error fetching {asset} ({timeframe}), attempt {attempt}/{max_retries}: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Network error fetching {asset} ({timeframe}) after {max_retries} attempts: {e}")
                        raise

                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error fetching {asset} ({timeframe}): {e}")
                    raise

                except Exception as e:
                    logger.error(f"Unexpected error fetching {asset} ({timeframe}): {e}")
                    raise

        # Store only the most granular timeframe in price_data
        if all_data:
            # Sort by timeframe duration
            all_data.sort(key=lambda x: timeframe_to_seconds(x[0]))

            # Store the most granular timeframe data
            self.price_data[asset] = all_data[0][1]

            logger.info(f"Price data initialized for {asset} with {len(self.price_data[asset])} samples")
        else:
            raise ValueError(f"No data could be fetched for {asset}")

    def _initialize_volatility_data(self, asset):
        """
        Initialize volatility data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data:
            logger.warning(f"Price data not initialized for {asset}")
            return

        price_data = self.price_data[asset]

        # Сначала создаем возвраты, так как они отсутствуют
        if 'log_return' not in price_data.columns:
            # Рассчитываем логарифмические возвраты
            price_data['log_return'] = np.log(price_data['close'] / price_data['close'].shift(1))
            # Заполняем первое значение нулем
            price_data.loc[price_data.index[0], 'log_return'] = 0

        # Calculate different volatility measures
        micro_vol_df = self.micro_analyzer.calculate_historical_volatility(price_data)
        meso_vol_df = self.meso_analyzer.calculate_historical_volatility(price_data)
        macro_vol_df = self.macro_analyzer.calculate_historical_volatility(price_data)

        # Create volatility DataFrame
        vol_data = pd.DataFrame(index=price_data.index)

        # Выбираем одну колонку волатильности из каждого DataFrame
        # Используем волатильность с окном среднего размера
        micro_vol_col = f'volatility_{self.micro_analyzer.window_sizes[2]}'
        meso_vol_col = f'volatility_{self.meso_analyzer.window_sizes[2]}'
        macro_vol_col = f'volatility_{self.macro_analyzer.window_sizes[2]}'

        vol_data['micro_vol'] = micro_vol_df[micro_vol_col] if micro_vol_col in micro_vol_df.columns else None
        vol_data['meso_vol'] = meso_vol_df[meso_vol_col] if meso_vol_col in meso_vol_df.columns else None
        vol_data['macro_vol'] = macro_vol_df[macro_vol_col] if macro_vol_col in macro_vol_df.columns else None

        # Calculate composite volatility
        weights = self.config.get('volatility_weights', {
            'micro': 0.3,
            'meso': 0.5,
            'macro': 0.2
        })

        # Убедимся, что все колонки существуют перед расчетом композитной волатильности
        if all(col in vol_data.columns for col in ['micro_vol', 'meso_vol', 'macro_vol']):
            vol_data['composite_vol'] = (
                    weights['micro'] * vol_data['micro_vol'] +
                    weights['meso'] * vol_data['meso_vol'] +
                    weights['macro'] * vol_data['macro_vol']
            )

        # Calculate fractal metrics
        try:
            # Calculate on rolling windows
            window_size = self.config.get('fractal_window', 100)
            # Убедимся, что значения положительные для логарифма
            close_positive = np.maximum(price_data['close'], 1e-10)
            returns = np.log(close_positive).diff().dropna()

            # Проверка, что у нас достаточно данных
            if len(returns) < window_size:
                logger.warning(f"Not enough data for fractal analysis: {len(returns)} < {window_size}")
                vol_data['hurst_exponent'] = np.nan
                vol_data['fractal_dimension'] = np.nan
            else:
                hurst_series = []
                fractal_dim_series = []

                # Use smaller step size for efficiency
                step_size = max(1, window_size // 10)

                for i in range(window_size, len(returns) + 1, step_size):
                    window = returns.iloc[i - window_size:i]

                    # Проверка на валидные данные
                    if window.isnull().any() or len(window) < window_size:
                        hurst = np.nan
                        fractal_dim = np.nan
                    else:
                        try:
                            hurst = calculate_hurst_exponent(window)
                            fractal_dim = calculate_fractal_dimension(window)
                        except Exception as calc_error:
                            logger.warning(f"Error calculating fractal metrics: {calc_error}")
                            hurst = np.nan
                            fractal_dim = np.nan

                    # Repeat value for all steps
                    hurst_series.extend([hurst] * min(step_size, len(returns) - i + window_size))
                    fractal_dim_series.extend([fractal_dim] * min(step_size, len(returns) - i + window_size))

                # Adjust list length if needed
                if len(hurst_series) > len(returns):
                    hurst_series = hurst_series[:len(returns)]
                    fractal_dim_series = fractal_dim_series[:len(returns)]
                elif len(hurst_series) < len(returns):
                    padding = [np.nan] * (len(returns) - len(hurst_series))
                    hurst_series = padding + hurst_series
                    fractal_dim_series = padding + fractal_dim_series

                # Add to DataFrame with same index as returns
                vol_data['hurst_exponent'] = pd.Series(hurst_series, index=returns.index)
                vol_data['fractal_dimension'] = pd.Series(fractal_dim_series, index=returns.index)

                # Repeat value for all steps
                hurst_series.extend([hurst] * min(step_size, len(returns) - i + window_size))
                fractal_dim_series.extend([fractal_dim] * min(step_size, len(returns) - i + window_size))

            # Adjust list length if needed
            if len(hurst_series) > len(returns):
                hurst_series = hurst_series[:len(returns)]
                fractal_dim_series = fractal_dim_series[:len(returns)]
            elif len(hurst_series) < len(returns):
                padding = [np.nan] * (len(returns) - len(hurst_series))
                hurst_series = padding + hurst_series
                fractal_dim_series = padding + fractal_dim_series

            # Add to DataFrame with same index as returns
            vol_data['hurst_exponent'] = pd.Series(hurst_series, index=returns.index)
            vol_data['fractal_dimension'] = pd.Series(fractal_dim_series, index=returns.index)

        except Exception as e:
            logger.warning(f"Error calculating fractal metrics for {asset}: {str(e)}")
            vol_data['hurst_exponent'] = np.nan
            vol_data['fractal_dimension'] = np.nan

        # Store the volatility data
        self.volatility_data[asset] = vol_data

        logger.info(f"Volatility data initialized for {asset}")

    def _initialize_kama(self, asset):
        """
        Initialize KAMA indicator for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data:
            raise ValueError(f"Price data not initialized for {asset}")

        # Get close prices
        close_prices = self.price_data[asset]['close']

        # Create KAMA instance
        kama = KAMAIndicator(
            er_period=self.kama_params.get('er_period', 10),
            fast_ef=self.kama_params.get('fast_ef', 0.666),
            slow_ef=self.kama_params.get('slow_ef', 0.0645)
        )

        # Calculate KAMA values
        kama_result = kama.calculate(pd.DataFrame({'close': close_prices}))
        kama_values = kama_result['kama'] if 'kama' in kama_result.columns else None

        # Store the KAMA instance and values
        self.kama[asset] = {
            'instance': kama,
            'values': kama_values
        }

        logger.info(f"KAMA initialized for {asset}")

    def _initialize_regime_data(self, asset):
        """
        Initialize market regime data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data or asset not in self.volatility_data:
            raise ValueError(f"Price and volatility data must be initialized for {asset}")

        price_data = self.price_data[asset]
        vol_data = self.volatility_data[asset]

        # Detect market regimes using the correct method

        regimes = self.regime_detector.fit_predict(
            price_data,
            price_col='close',
            vol_col='composite_vol' if 'composite_vol' in vol_data.columns else None
        )

        # Store regime data
        self.regime_data[asset] = regimes

        logger.info(f"Regime data initialized for {asset}")

    def _initialize_predictions(self, asset):
        """
        Initialize volatility predictions for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if not self.use_ml or not self.prediction_model:
            return

        if asset not in self.price_data or asset not in self.volatility_data:
            raise ValueError(f"Price and volatility data must be initialized for {asset}")

        # Initialize the model if not done already
        if not hasattr(self.prediction_model, 'models') or not self.prediction_model.models:
            self.prediction_model.initialize_models()

        # Prepare data for training
        price_data = self.price_data[asset]
        vol_data = self.volatility_data[asset]
        regime_data = self.regime_data.get(asset, None)

        # Combined data for training - используйте переименование колонок, чтобы избежать конфликта
        combined_data = price_data[['close']].copy().rename(columns={'close': 'price_close'})
        combined_data = combined_data.join(vol_data)

        if regime_data is not None:
            # Добавьте суффиксы, чтобы избежать конфликта колонок
            combined_data = combined_data.join(regime_data, lsuffix='_vol', rsuffix='_regime')

        # Clean up missing values
        combined_data = combined_data.dropna()

        if len(combined_data) < 100:
            logger.warning(f"Not enough data to train prediction model for {asset}")
            return

        # Train the model
        logger.info(f"Training prediction model for {asset}")
        self.prediction_model.train(combined_data, target_column='composite_vol')

        # Make predictions
        predictions = self.prediction_model.predict(combined_data)

        # Store predictions
        self.predictions[asset] = predictions

        logger.info(f"Predictions initialized for {asset}")

    def _initialize_signals(self, asset):
        """
        Initialize trading signals for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data or asset not in self.volatility_data:
            raise ValueError(f"Price and volatility data must be initialized for {asset}")

        price_data = self.price_data[asset]
        vol_data = self.volatility_data[asset]
        regime_data = self.regime_data.get(asset, None)
        predictions = self.predictions.get(asset, None)

        # Generate signals
        signals = self.signal_generator.generate_signals(
            price_data=price_data,
            volatility_data=vol_data,
            regime_data=regime_data
        )

        # Store signals
        self.signals[asset] = signals

        logger.info(f"Signals initialized for {asset}")

    def update(self):
        """Update all data for all assets."""
        if not self.initialized:
            logger.warning("Indicator not initialized. Call initialize_assets() first.")
            return False

        for asset in self.assets:
            try:
                logger.debug(f"Updating data for {asset}")

                # Update price data
                self._update_price_data(asset)

                # Update volatility data
                self._update_volatility_data(asset)

                # Update KAMA
                self._update_kama(asset)

                # Update regime data
                self._update_regime_data(asset)

                # Update predictions
                if self.use_ml and self.prediction_model:
                    self._update_predictions(asset)

                # Update signals
                self._update_signals(asset)

            except Exception as e:
                logger.error(f"Error updating {asset}: {str(e)}")
                continue

        logger.info("Update completed for all assets")
        return True

    def _update_price_data(self, asset):
        """
        Update price data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data:
            logger.warning(f"Price data not initialized for {asset}")
            return

        # Get current data
        current_data = self.price_data[asset]

        # Get the most recent timestamp
        last_timestamp = current_data.index[-1]

        # Get timeframe from first entry in timeframes
        timeframe = self.timeframes[0]

        try:
            # Fetch new data since last timestamp
            since = int(last_timestamp.timestamp() * 1000)

            # Add a small buffer to avoid duplicate data
            since += 1

            ohlcv = self.exchange.fetch_ohlcv(
                symbol=asset,
                timeframe=timeframe,
                since=since,
                limit=100  # Limit to recent data
            )

            if not ohlcv:
                logger.debug(f"No new data for {asset}")
                return

            # Convert to DataFrame
            new_data = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
            new_data.set_index('timestamp', inplace=True)

            # Check for duplicate timestamps
            new_data = new_data[~new_data.index.isin(current_data.index)]

            if new_data.empty:
                logger.debug(f"No new data for {asset}")
                return

            # Append new data
            updated_data = pd.concat([current_data, new_data])

            # Store updated data
            self.price_data[asset] = updated_data

            logger.debug(f"Added {len(new_data)} new candlesticks for {asset}")

        except Exception as e:
            logger.error(f"Error updating price data for {asset}: {str(e)}")
            raise

    def _update_volatility_data(self, asset):
        """
        Update volatility data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data:
            logger.warning(f"Price data not initialized for {asset}")
            return

        if asset not in self.volatility_data:
            logger.warning(f"Volatility data not initialized for {asset}")
            return

        price_data = self.price_data[asset]
        current_vol_data = self.volatility_data[asset]

        # Get the timeframe of the data
        price_index = price_data.index
        if len(price_index) < 2:
            logger.warning(f"Not enough price data for {asset}")
            return

        # Check if we have new price data
        if price_data.index[-1] <= current_vol_data.index[-1]:
            logger.debug(f"No new price data for {asset}")
            return

        # Calculate new volatility measures
        micro_vol = self.micro_analyzer.calculate_historical_volatility(price_data)
        meso_vol = self.meso_analyzer.calculate_historical_volatility(price_data)
        macro_vol = self.macro_analyzer.calculate_historical_volatility(price_data)

        # Create new volatility DataFrame
        new_vol_data = pd.DataFrame(index=price_data.index)
        new_vol_data['micro_vol'] = micro_vol
        new_vol_data['meso_vol'] = meso_vol
        new_vol_data['macro_vol'] = macro_vol

        # Calculate composite volatility
        weights = self.config.get('volatility_weights', {
            'micro': 0.3,
            'meso': 0.5,
            'macro': 0.2
        })

        new_vol_data['composite_vol'] = (
                weights['micro'] * new_vol_data['micro_vol'] +
                weights['meso'] * new_vol_data['meso_vol'] +
                weights['macro'] * new_vol_data['macro_vol']
        )

        # Calculate fractal metrics for new data only
        try:
            # Get new data points
            new_indices = new_vol_data.index[~new_vol_data.index.isin(current_vol_data.index)]

            if len(new_indices) > 0:
                window_size = self.config.get('fractal_window', 100)
                returns = np.log(price_data['close']).diff().dropna()

                # Calculate for each new point
                for idx in new_indices:
                    if idx in returns.index:
                        # Get window ending at current index
                        i = returns.index.get_loc(idx)

                        if i >= window_size:
                            window = returns.iloc[i - window_size:i + 1]

                            hurst = calculate_hurst_exponent(window)
                            fractal_dim = calculate_fractal_dimension(window)

                            new_vol_data.loc[idx, 'hurst_exponent'] = hurst
                            new_vol_data.loc[idx, 'fractal_dimension'] = fractal_dim

        except Exception as e:
            logger.warning(f"Error calculating fractal metrics for {asset}: {str(e)}")

        # Combine old and new data
        combined_vol_data = pd.concat([current_vol_data, new_vol_data])

        # Remove duplicates if any
        combined_vol_data = combined_vol_data[~combined_vol_data.index.duplicated(keep='last')]

        # Store updated volatility data
        self.volatility_data[asset] = combined_vol_data

        logger.debug(f"Volatility data updated for {asset}")

    def _update_kama(self, asset):
        """
        Update KAMA indicator for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data:
            logger.warning(f"Price data not initialized for {asset}")
            return

        if asset not in self.kama:
            logger.warning(f"KAMA not initialized for {asset}")
            return

        # Get close prices
        close_prices = self.price_data[asset]['close']

        # Get KAMA instance
        kama_instance = self.kama[asset]['instance']

        # Update KAMA values
        kama_values = kama_instance.calculate(close_prices)

        # Store updated values
        self.kama[asset]['values'] = kama_values

        logger.debug(f"KAMA updated for {asset}")

    def _update_regime_data(self, asset):
        """
        Update market regime data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data or asset not in self.volatility_data:
            logger.warning(f"Price and volatility data must be initialized for {asset}")
            return

        if asset not in self.regime_data:
            logger.warning(f"Regime data not initialized for {asset}")
            return

        price_data = self.price_data[asset]
        vol_data = self.volatility_data[asset]
        current_regime_data = self.regime_data[asset]

        # Check if we have new data
        if price_data.index[-1] <= current_regime_data.index[-1]:
            logger.debug(f"No new data for regime update for {asset}")
            return

        # Detect market regimes using fit_predict instead of detect_regimes
        new_regimes = self.regime_detector.fit_predict(
            price_data,
            price_col='close',
            vol_col='composite_vol' if 'composite_vol' in vol_data.columns else None
        )

        # Combine old and new data
        combined_regimes = pd.concat([current_regime_data, new_regimes])

        # Remove duplicates if any
        combined_regimes = combined_regimes[~combined_regimes.index.duplicated(keep='last')]

        # Store updated regime data
        self.regime_data[asset] = combined_regimes

        logger.debug(f"Regime data updated for {asset}")

    def _update_predictions(self, asset):
        """
        Update volatility predictions for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if not self.use_ml or not self.prediction_model:
            return

        if asset not in self.price_data or asset not in self.volatility_data:
            logger.warning(f"Price and volatility data must be initialized for {asset}")
            return

        # Prepare data for prediction
        price_data = self.price_data[asset]
        vol_data = self.volatility_data[asset]
        regime_data = self.regime_data.get(asset, None)

        # Combined data
        combined_data = price_data[['close']].copy()
        combined_data = combined_data.join(vol_data)

        if regime_data is not None:
            combined_data = combined_data.join(regime_data)

        # Clean up missing values
        combined_data = combined_data.dropna()

        # Get current regime if available
        current_regime = None
        if regime_data is not None and 'regime' in regime_data.columns:
            current_regime = regime_data['regime'].iloc[-1]

        # Make predictions
        predictions = self.prediction_model.predict(combined_data, regime=current_regime)

        # Store predictions
        self.predictions[asset] = predictions

        logger.debug(f"Predictions updated for {asset}")

    def _update_signals(self, asset):
        """
        Update trading signals for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        """
        if asset not in self.price_data or asset not in self.volatility_data:
            logger.warning(f"Price and volatility data must be initialized for {asset}")
            return

        price_data = self.price_data[asset]
        vol_data = self.volatility_data[asset]
        regime_data = self.regime_data.get(asset, None)
        predictions = self.predictions.get(asset, None)

        # Generate signals
        signals = self.signal_generator.generate_signals(
            price_data=price_data,
            volatility_data=vol_data,
            regime_data=regime_data
        )

        # Store signals
        self.signals[asset] = signals

        logger.debug(f"Signals updated for {asset}")

    def start_background_update(self, interval=None):
        """
        Start background thread for automatic updates.

        Parameters:
        -----------
        interval : int, optional
            Update interval in seconds
        """
        if self.update_thread_active:
            logger.warning("Background update thread is already running")
            return

        if interval is not None:
            self.update_interval = interval

        # Start thread
        self.update_thread_active = True
        self.update_thread = threading.Thread(target=self._background_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        logger.info(f"Background update thread started with interval {self.update_interval}s")

    def stop_background_update(self):
        """Stop background update thread."""
        if not self.update_thread_active:
            logger.warning("Background update thread is not running")
            return

        self.update_thread_active = False

        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)

        logger.info("Background update thread stopped")

    def _background_update_loop(self):
        """Background loop for automatic updates."""
        while self.update_thread_active:
            try:
                self.update()
            except Exception as e:
                logger.error(f"Error in background update: {str(e)}")

            # Sleep until next update
            time.sleep(self.update_interval)

    def get_volatility_data(self, asset=None, start_date=None, end_date=None):
        """
        Get volatility data for an asset.

        Parameters:
        -----------
        asset : str, optional
            Asset symbol. If None, returns data for first asset.
        start_date : datetime, optional
            Start date for filtering
        end_date : datetime, optional
            End date for filtering

        Returns:
        --------
        pd.DataFrame
            Volatility data
        """
        if asset is None and self.assets:
            asset = self.assets[0]

        if asset not in self.volatility_data:
            logger.warning(f"No volatility data for {asset}")
            return None

        data = self.volatility_data[asset]

        # Apply date filters if specified
        if start_date is not None:
            data = data[data.index >= start_date]

        if end_date is not None:
            data = data[data.index <= end_date]

        return data

    def get_price_data(self, asset=None, start_date=None, end_date=None):
        """
        Get price data for an asset.

        Parameters:
        -----------
        asset : str, optional
            Asset symbol. If None, returns data for first asset.
        start_date : datetime, optional
            Start date for filtering
        end_date : datetime, optional
            End date for filtering

        Returns:
        --------
        pd.DataFrame
            Price data
        """
        if asset is None and self.assets:
            asset = self.assets[0]

        if asset not in self.price_data:
            logger.warning(f"No price data for {asset}")
            return None

        data = self.price_data[asset]

        # Apply date filters if specified
        if start_date is not None:
            data = data[data.index >= start_date]

        if end_date is not None:
            data = data[data.index <= end_date]

        return data

    def get_regime_data(self, asset=None, start_date=None, end_date=None):
        """
        Get regime data for an asset.

        Parameters:
        -----------
        asset : str, optional
            Asset symbol. If None, returns data for first asset.
        start_date : datetime, optional
            Start date for filtering
        end_date : datetime, optional
            End date for filtering

        Returns:
        --------
        pd.DataFrame
            Regime data
        """
        if asset is None and self.assets:
            asset = self.assets[0]

        if asset not in self.regime_data:
            logger.warning(f"No regime data for {asset}")
            return None

        data = self.regime_data[asset]

        # Apply date filters if specified
        if start_date is not None:
            data = data[data.index >= start_date]

        if end_date is not None:
            data = data[data.index <= end_date]

        return data

    def get_predictions(self, asset=None, start_date=None, end_date=None):
        """
        Get volatility predictions for an asset.

        Parameters:
        -----------
        asset : str, optional
            Asset symbol. If None, returns data for first asset.
        start_date : datetime, optional
            Start date for filtering
        end_date : datetime, optional
            End date for filtering

        Returns:
        --------
        pd.Series
            Predicted volatility
        """
        if not self.use_ml or not self.prediction_model:
            return None

        if asset is None and self.assets:
            asset = self.assets[0]

        if asset not in self.predictions:
            logger.warning(f"No predictions for {asset}")
            return None

        data = self.predictions[asset]

        # Apply date filters if specified
        if start_date is not None:
            data = data[data.index >= start_date]

        if end_date is not None:
            data = data[data.index <= end_date]

        return data

    def get_signal_data(self, asset=None, start_date=None, end_date=None):
        """
        Get signal data for an asset.

        Parameters:
        -----------
        asset : str, optional
            Asset symbol. If None, returns data for first asset.
        start_date : datetime, optional
            Start date for filtering
        end_date : datetime, optional
            End date for filtering

        Returns:
        --------
        pd.DataFrame
            Signal data
        """
        if asset is None and self.assets:
            asset = self.assets[0]

        if asset not in self.signals:
            logger.warning(f"No signals for {asset}")
            return None

        data = self.signals[asset]

        # Apply date filters if specified
        if start_date is not None:
            data = data[data.index >= start_date]

        if end_date is not None:
            data = data[data.index <= end_date]

        return data

    def get_monitored_assets(self):
        """
        Get list of all monitored assets.

        Returns:
        --------
        list
            List of asset symbols
        """
        return self.assets

    def is_valid_asset(self, asset):
        """
        Check if an asset is valid and monitored.

        Parameters:
        -----------
        asset : str
            Asset symbol

        Returns:
        --------
        bool
            True if asset is valid
        """
        return asset in self.assets

    def save_data(self, directory=None):
        """
        Save all data to files.

        Parameters:
        -----------
        directory : str, optional
            Directory to save data. If None, uses config data_dir.

        Returns:
        --------
        bool
            True if successful
        """
        if directory is None:
            directory = self.data_dir

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save data for each asset
        for asset in self.assets:
            asset_dir = os.path.join(directory, asset.replace('/', '_'))
            os.makedirs(asset_dir, exist_ok=True)

            # Save price data
            if asset in self.price_data:
                price_file = os.path.join(asset_dir, 'price_data.csv')
                self.price_data[asset].to_csv(price_file)

            # Save volatility data
            if asset in self.volatility_data:
                vol_file = os.path.join(asset_dir, 'volatility_data.csv')
                self.volatility_data[asset].to_csv(vol_file)

            # Save regime data
            if asset in self.regime_data:
                regime_file = os.path.join(asset_dir, 'regime_data.csv')
                self.regime_data[asset].to_csv(regime_file)

            # Save predictions
            if asset in self.predictions:
                pred_file = os.path.join(asset_dir, 'predictions.csv')
                self.predictions[asset].to_csv(pred_file)

            # Save signals
            if asset in self.signals:
                signal_file = os.path.join(asset_dir, 'signals.csv')
                self.signals[asset].to_csv(signal_file)

        logger.info(f"All data saved to {directory}")
        return True

    def load_data(self, directory=None):
        """
        Load all data from files.

        Parameters:
        -----------
        directory : str, optional
            Directory to load data from. If None, uses config data_dir.

        Returns:
        --------
        bool
            True if successful
        """
        if directory is None:
            directory = self.data_dir

        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return False

        # Load data for each asset
        for asset in self.assets:
            asset_dir = os.path.join(directory, asset.replace('/', '_'))

            if not os.path.exists(asset_dir):
                logger.warning(f"No data directory for {asset}")
                continue

            try:
                # Load price data
                price_file = os.path.join(asset_dir, 'price_data.csv')
                if os.path.exists(price_file):
                    self.price_data[asset] = pd.read_csv(price_file, index_col=0, parse_dates=True)

                # Load volatility data
                vol_file = os.path.join(asset_dir, 'volatility_data.csv')
                if os.path.exists(vol_file):
                    self.volatility_data[asset] = pd.read_csv(vol_file, index_col=0, parse_dates=True)

                # Load regime data
                regime_file = os.path.join(asset_dir, 'regime_data.csv')
                if os.path.exists(regime_file):
                    self.regime_data[asset] = pd.read_csv(regime_file, index_col=0, parse_dates=True)

                # Load predictions
                pred_file = os.path.join(asset_dir, 'predictions.csv')
                if os.path.exists(pred_file):
                    self.predictions[asset] = pd.read_csv(pred_file, index_col=0, parse_dates=True).iloc[:, 0]

                # Load signals
                signal_file = os.path.join(asset_dir, 'signals.csv')
                if os.path.exists(signal_file):
                    self.signals[asset] = pd.read_csv(signal_file, index_col=0, parse_dates=True)

                logger.info(f"Data loaded for {asset}")

            except Exception as e:
                logger.error(f"Error loading data for {asset}: {str(e)}")
                continue

        # Set initialization flag
        self.initialized = True

        # Connect signal generator to this indicator
        self.signal_generator.set_volatility_indicator(self)

        logger.info(f"All data loaded from {directory}")
        return True

    def add_asset(self, asset):
        """
        Add a new asset to monitor.

        Parameters:
        -----------
        asset : str
            Asset symbol

        Returns:
        --------
        bool
            True if successful
        """
        if asset in self.assets:
            logger.warning(f"Asset {asset} is already monitored")
            return False

        try:
            # Add to assets list
            self.assets.append(asset)

            # Initialize data for new asset
            logger.info(f"Initializing data for {asset}")

            # Fetch initial price data
            self._initialize_price_data(asset)

            # Calculate initial volatility data
            self._initialize_volatility_data(asset)

            # Initialize KAMA
            self._initialize_kama(asset)

            # Initialize regime data
            self._initialize_regime_data(asset)

            # Initialize prediction model
            if self.use_ml and self.prediction_model:
                self._initialize_predictions(asset)

            # Initialize signals
            self._initialize_signals(asset)

            logger.info(f"Asset {asset} added successfully")
            return True

        except Exception as e:
            logger.error(f"Error adding asset {asset}: {str(e)}")

            # Remove from assets list if added
            if asset in self.assets:
                self.assets.remove(asset)

            return False

    def remove_asset(self, asset):
        """
        Remove an asset from monitoring.

        Parameters:
        -----------
        asset : str
            Asset symbol

        Returns:
        --------
        bool
            True if successful
        """
        if asset not in self.assets:
            logger.warning(f"Asset {asset} is not monitored")
            return False

        # Remove from assets list
        self.assets.remove(asset)

        # Remove data
        if asset in self.price_data:
            del self.price_data[asset]

        if asset in self.volatility_data:
            del self.volatility_data[asset]

        if asset in self.kama:
            del self.kama[asset]

        if asset in self.regime_data:
            del self.regime_data[asset]

        if asset in self.predictions:
            del self.predictions[asset]

        if asset in self.signals:
            del self.signals[asset]

        logger.info(f"Asset {asset} removed successfully")
        return True

    def fetch_blockchain_metrics(self, asset=None):
        """
        Fetch blockchain metrics for an asset.

        Parameters:
        -----------
        asset : str, optional
            Asset symbol

        Returns:
        --------
        pd.DataFrame
            Blockchain metrics data
        """
        if asset is None and self.assets:
            asset = self.assets[0]

        # Placeholder for actual blockchain data fetching
        logger.info(f"Fetching blockchain metrics for {asset}")

        # Here you would integrate with blockchain data providers
        # For now, return empty DataFrame
        return pd.DataFrame()

    def integrate_network_metrics(self, asset=None):
        """
        Integrate network metrics with volatility data.

        Parameters:
        -----------
        asset : str, optional
            Asset symbol

        Returns:
        --------
        pd.DataFrame
            Integrated data
        """
        if asset is None and self.assets:
            asset = self.assets[0]

        if asset not in self.volatility_data:
            logger.warning(f"No volatility data for {asset}")
            return None

        # Fetch blockchain metrics
        blockchain_data = self.fetch_blockchain_metrics(asset)

        if blockchain_data.empty:
            logger.warning(f"No blockchain data available for {asset}")
            return self.volatility_data[asset]

        # Merge with volatility data
        vol_data = self.volatility_data[asset]

        # Here you would perform the actual integration
        # For now, just return the volatility data
        return vol_data

    def calculate_stop_loss(self, price_data=None, volatility_data=None, position_type='long', asset=None):
        """
        Calculate dynamic stop loss levels.

        Parameters:
        -----------
        price_data : pd.DataFrame, optional
            Price data
        volatility_data : pd.DataFrame, optional
            Volatility data
        position_type : str
            'long' or 'short'
        asset : str, optional
            Asset symbol

        Returns:
        --------
        pd.Series
            Stop loss levels
        """
        # If no explicit data provided, use data for specified asset
        if price_data is None or volatility_data is None:
            if asset is None and self.assets:
                asset = self.assets[0]

            if asset not in self.price_data or asset not in self.volatility_data:
                logger.warning(f"No data for {asset}")
                return None

            price_data = self.price_data[asset]
            volatility_data = self.volatility_data[asset]

        # Use signal generator to calculate stop loss
        return self.signal_generator.calculate_stop_loss(
            price_data=price_data,
            volatility_data=volatility_data,
            position_type=position_type
        )

    def __str__(self):
        """String representation of the indicator."""
        assets_str = ", ".join(self.assets)
        status = "Initialized" if self.initialized else "Not initialized"

        return f"Progressive Adaptive Volatility Indicator (status: {status}, assets: {assets_str})"

    def __repr__(self):
        """Detailed representation of the indicator."""
        return self.__str__() + f" at {hex(id(self))}"

    def calculate_take_profit(self, price_data=None, volatility_data=None, position_type='long', asset=None):
        """
        Calculate dynamic take profit levels.

        Parameters:
        -----------
        price_data : pd.DataFrame, optional
            Price data
        volatility_data : pd.DataFrame, optional
            Volatility data
        position_type : str
            'long' or 'short'
        asset : str, optional
            Asset symbol

        Returns:
        --------
        pd.Series
            Take profit levels
        """
        # If no explicit data provided, use data for specified asset
        if price_data is None or volatility_data is None:
            if asset is None and self.assets:
                asset = self.assets[0]

            if asset not in self.price_data or asset not in self.volatility_data:
                logger.warning(f"No data for {asset}")
                return None

            price_data = self.price_data[asset]
            volatility_data = self.volatility_data[asset]

        # Use signal generator to calculate take profit
        return self.signal_generator.calculate_take_profit(
            price_data=price_data,
            volatility_data=volatility_data,
            position_type=position_type
        )