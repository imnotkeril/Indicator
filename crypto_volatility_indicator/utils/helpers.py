"""
Helper functions for the Crypto Volatility Indicator.
Contains various utility functions used throughout the project.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import hashlib
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger
# Time and date utilities
def timestamp_to_datetime(timestamp, unit='ms'):
    """
    Convert timestamp to datetime.

    Parameters:
    -----------
    timestamp : int
        Unix timestamp
    unit : str
        Unit of timestamp ('ms' or 's')

    Returns:
    --------
    datetime
        Converted datetime
    """
    if unit == 'ms':
        return datetime.fromtimestamp(timestamp / 1000.0)
    else:
        return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt, unit='ms'):
    """
    Convert datetime to timestamp.

    Parameters:
    -----------
    dt : datetime
        Datetime to convert
    unit : str
        Unit of timestamp ('ms' or 's')

    Returns:
    --------
    int
        Unix timestamp
    """
    if unit == 'ms':
        return int(dt.timestamp() * 1000)
    else:
        return int(dt.timestamp())


def timeframe_to_seconds(timeframe):
    """
    Convert timeframe string to seconds.

    Parameters:
    -----------
    timeframe : str
        Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
    --------
    int
        Seconds
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])

    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 60 * 60
    elif unit == 'd':
        return value * 24 * 60 * 60
    elif unit == 'w':
        return value * 7 * 24 * 60 * 60
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")


def timeframe_to_timedelta(timeframe):
    """
    Convert timeframe string to timedelta.

    Parameters:
    -----------
    timeframe : str
        Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
    --------
    timedelta
        Timedelta
    """
    seconds = timeframe_to_seconds(timeframe)
    return timedelta(seconds=seconds)


def resample_dataframe(df, timeframe, price_col='close', time_col='timestamp'):
    """
    Resample DataFrame to a different timeframe.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to resample
    timeframe : str
        Target timeframe (e.g., '1h', '4h', '1d')
    price_col : str
        Column name for price data
    time_col : str
        Column name for timestamp data

    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col in df.columns:
            df = df.set_index(time_col)
        else:
            raise ValueError(f"DataFrame has no timestamp column: {time_col}")

    # Convert timeframe to pandas frequency string
    if timeframe[-1] == 'm':
        freq = f"{timeframe[:-1]}T"
    elif timeframe[-1] == 'h':
        freq = f"{timeframe[:-1]}H"
    elif timeframe[-1] == 'd':
        freq = f"{timeframe[:-1]}D"
    elif timeframe[-1] == 'w':
        freq = f"{timeframe[:-1]}W"
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    # Perform resampling
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    return resampled


# Data handling utilities
def calculate_returns(prices, method='log'):
    """
    Calculate returns from price series.

    Parameters:
    -----------
    prices : array-like
        Price series
    method : str
        Method to calculate returns ('log' or 'pct')

    Returns:
    --------
    array-like
        Returns
    """
    if method == 'log':
        return np.log(prices / np.roll(prices, 1))[1:]
    elif method == 'pct':
        return (prices / np.roll(prices, 1) - 1)[1:]
    else:
        raise ValueError(f"Unknown return calculation method: {method}")


def calculate_volatility(returns, window, annualize=True):
    """
    Calculate volatility from returns.

    Parameters:
    -----------
    returns : array-like
        Returns series
    window : int
        Rolling window size
    annualize : bool
        Whether to annualize the volatility

    Returns:
    --------
    array-like
        Volatility
    """
    # Convert to numpy array if needed
    if isinstance(returns, list):
        returns = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.values

    # Calculate standard deviation
    if len(returns) < window:
        return np.nan

    vol = np.std(returns, ddof=1)

    # Annualize if requested
    if annualize:
        vol *= np.sqrt(252)  # Assuming 252 trading days in a year

    return vol


def calculate_rolling_volatility(returns, window, annualize=True):
    """
    Calculate rolling volatility from returns.

    Parameters:
    -----------
    returns : array-like
        Returns series
    window : int
        Rolling window size
    annualize : bool
        Whether to annualize the volatility

    Returns:
    --------
    array-like
        Rolling volatility
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    vol = returns.rolling(window=window).std(ddof=1)

    # Annualize if requested
    if annualize:
        vol *= np.sqrt(252)  # Assuming 252 trading days in a year

    return vol


# File handling utilities
def ensure_directory(path):
    """
    Ensure a directory exists.

    Parameters:
    -----------
    path : str or Path
        Directory path

    Returns:
    --------
    Path
        Directory path
    """
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path


def save_dataframe(df, path, format='csv'):
    """
    Save DataFrame to file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    path : str or Path
        File path
    format : str
        File format ('csv', 'pickle', 'parquet')

    Returns:
    --------
    Path
        File path
    """
    path = Path(path)
    ensure_directory(path.parent)

    if format == 'csv':
        df.to_csv(path)
    elif format == 'pickle':
        df.to_pickle(path)
    elif format == 'parquet':
        df.to_parquet(path)
    else:
        raise ValueError(f"Unknown format: {format}")

    return path


def load_dataframe(path, format=None):
    """
    Load DataFrame from file.

    Parameters:
    -----------
    path : str or Path
        File path
    format : str, optional
        File format ('csv', 'pickle', 'parquet')
        If None, inferred from file extension

    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if format is None:
        # Infer format from extension
        suffix = path.suffix.lower()
        if suffix == '.csv':
            format = 'csv'
        elif suffix in ['.pkl', '.pickle']:
            format = 'pickle'
        elif suffix == '.parquet':
            format = 'parquet'
        else:
            raise ValueError(f"Unknown file extension: {suffix}")

    if format == 'csv':
        return pd.read_csv(path)
    elif format == 'pickle':
        return pd.read_pickle(path)
    elif format == 'parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unknown format: {format}")


# Cache utilities
def create_cache_key(*args, **kwargs):
    """
    Create a cache key from arguments.

    Parameters:
    -----------
    *args : any
        Positional arguments
    **kwargs : any
        Keyword arguments

    Returns:
    --------
    str
        Cache key
    """
    key_data = {'args': args, 'kwargs': {k: v for k, v in kwargs.items() if
                                         isinstance(v, (str, int, float, bool, list, tuple, dict))}}
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


class SimpleCache:
    """Simple in-memory cache for function results."""

    def __init__(self, max_size=1000, ttl=None):
        """
        Initialize cache.

        Parameters:
        -----------
        max_size : int
            Maximum number of items to store
        ttl : int, optional
            Time-to-live in seconds
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key):
        """
        Get value from cache.

        Parameters:
        -----------
        key : str
            Cache key

        Returns:
        --------
        any
            Cached value or None if not found
        """
        if key not in self.cache:
            return None

        value, timestamp = self.cache[key]

        # Check if expired
        if self.ttl is not None and time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None

        return value

    def set(self, key, value):
        """
        Set value in cache.

        Parameters:
        -----------
        key : str
            Cache key
        value : any
            Value to cache
        """
        # Evict oldest items if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (value, time.time())

    def clear(self):
        """Clear the cache."""
        self.cache.clear()


def cached(cache=None, ttl=None):
    """
    Decorator for caching function results.

    Parameters:
    -----------
    cache : SimpleCache, optional
        Cache instance to use
    ttl : int, optional
        Time-to-live in seconds

    Returns:
    --------
    function
        Decorated function
    """
    if cache is None:
        cache = SimpleCache(ttl=ttl)

    def decorator(func):
        def wrapper(*args, **kwargs):
            key = create_cache_key(func.__name__, *args, **kwargs)
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
            return result
        return wrapper
    return decorator


# Statistical functions
def calculate_z_score(values, window=None):
    """
    Calculate z-score for a series of values.

    Parameters:
    -----------
    values : array-like
        Values series
    window : int, optional
        Rolling window size. If None, use the entire series.

    Returns:
    --------
    array-like
        Z-scores
    """
    if window is None:
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        return (values - mean) / std
    else:
        if isinstance(values, np.ndarray):
            values = pd.Series(values)

        rolling_mean = values.rolling(window=window).mean()
        rolling_std = values.rolling(window=window).std(ddof=1)

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        return (values - rolling_mean) / rolling_std


def calculate_percentile(values, window=None):
    """
    Calculate percentile rank for a series of values.

    Parameters:
    -----------
    values : array-like
        Values series
    window : int, optional
        Rolling window size. If None, use the entire series.

    Returns:
    --------
    array-like
        Percentile ranks (0-1)
    """
    if window is None:
        return pd.Series(values).rank(pct=True)
    else:
        if isinstance(values, np.ndarray):
            values = pd.Series(values)

        def rolling_percentile(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        return values.rolling(window=window).apply(rolling_percentile, raw=False)


def calculate_ewma(values, alpha=0.2):
    """
    Calculate exponentially weighted moving average.

    Parameters:
    -----------
    values : array-like
        Values series
    alpha : float
        Smoothing factor

    Returns:
    --------
    array-like
        EWMA
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    return values.ewm(alpha=alpha, adjust=False).mean()


def calculate_bollinger_bands(values, window=20, num_std=2):
    """
    Calculate Bollinger Bands.

    Parameters:
    -----------
    values : array-like
        Values series
    window : int
        Rolling window size
    num_std : float
        Number of standard deviations for bands

    Returns:
    --------
    tuple
        (middle_band, upper_band, lower_band)
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    middle_band = values.rolling(window=window).mean()
    std_dev = values.rolling(window=window).std(ddof=1)

    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    return middle_band, upper_band, lower_band


def calculate_macd(values, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Parameters:
    -----------
    values : array-like
        Values series
    fast_period : int
        Fast EMA period
    slow_period : int
        Slow EMA period
    signal_period : int
        Signal EMA period

    Returns:
    --------
    tuple
        (macd_line, signal_line, histogram)
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    fast_ema = values.ewm(span=fast_period, adjust=False).mean()
    slow_ema = values.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_rsi(values, window=14):
    """
    Calculate RSI (Relative Strength Index).

    Parameters:
    -----------
    values : array-like
        Values series
    window : int
        RSI period

    Returns:
    --------
    array-like
        RSI values (0-100)
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    # Calculate price changes
    delta = values.diff()

    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()

    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)

    # Calculate average gains and losses
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Time series decomposition
def decompose_time_series(values, period=None, method='additive'):
    """
    Decompose time series into trend, seasonal, and residual components.

    Parameters:
    -----------
    values : array-like
        Time series values
    period : int, optional
        Length of seasonal period
    method : str
        Decomposition method ('additive' or 'multiplicative')

    Returns:
    --------
    tuple
        (trend, seasonal, residual)
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    # Estimate period if not provided
    if period is None:
        # Try to detect weekly seasonality (7 days)
        if len(values) >= 14:
            period = 7
        else:
            period = len(values) // 2

    # Perform decomposition
    result = seasonal_decompose(values, model=method, period=period)

    return result.trend, result.seasonal, result.resid


# Fractal analysis
def calculate_hurst_exponent(values, max_lag=20):
    """
    Calculate Hurst exponent to measure long-term memory in time series.

    Parameters:
    -----------
    values : array-like
        Time series values
    max_lag : int
        Maximum lag for R/S analysis

    Returns:
    --------
    float
        Hurst exponent
    """
    if isinstance(values, pd.Series):
        values = values.values

    # Convert to numpy array if needed
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    # Calculate returns
    returns = np.diff(np.log(values))

    # Calculate R/S for different lags
    rs_values = []
    lags = range(2, min(max_lag, len(returns) // 2))

    for lag in lags:
        # Split returns into chunks
        chunk_size = len(returns) // lag
        chunks = [returns[i:i + chunk_size] for i in range(0, len(returns) - chunk_size + 1, chunk_size)]

        # Calculate R/S for each chunk
        rs = []
        for chunk in chunks:
            mean = np.mean(chunk)
            cumsum = np.cumsum(chunk - mean)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs.append(r / s)

        # Average R/S values
        if rs:
            rs_values.append(np.mean(rs))

    # Calculate Hurst exponent using linear regression
    if len(lags) > 1 and len(rs_values) > 1:
        x = np.log10(lags)
        y = np.log10(rs_values)

        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        return m
    else:
        return np.nan


def calculate_fractal_dimension(values, max_lag=20):
    """
    Calculate fractal dimension using the box-counting method.

    Parameters:
    -----------
    values : array-like
        Time series values
    max_lag : int
        Maximum lag for box-counting

    Returns:
    --------
    float
        Fractal dimension
    """
    if isinstance(values, pd.Series):
        values = values.values

    # Convert to numpy array if needed
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    # Normalize values to [0, 1]
    min_val = np.min(values)
    max_val = np.max(values)

    if max_val > min_val:
        norm_values = (values - min_val) / (max_val - min_val)
    else:
        return 1.0  # Constant series has dimension 1

    # Calculate box counts for different scales
    box_counts = []
    scales = []

    for k in range(1, max_lag + 1):
        scale = 1.0 / k
        count = 0

        for i in range(len(norm_values) - 1):
            # Count boxes needed to cover the curve
            box_x = int(i * scale)
            box_y = int(norm_values[i] * k)

            next_box_x = int((i + 1) * scale)
            next_box_y = int(norm_values[i + 1] * k)

            # If curve crosses to a new box, increment count
            if box_x != next_box_x or box_y != next_box_y:
                count += 1

        box_counts.append(count)
        scales.append(scale)

    # Calculate fractal dimension using linear regression
    if len(scales) > 1 and len(box_counts) > 1:
        x = np.log(scales)
        y = np.log(box_counts)

        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        return -m
    else:
        return 1.0  # Default to dimension 1 if calculation fails


# Volatility modeling
def calculate_garch_parameters(returns, p=1, q=1):
    """
    Estimate GARCH model parameters.

    Parameters:
    -----------
    returns : array-like
        Returns series
    p : int
        GARCH lag order
    q : int
        ARCH lag order

    Returns:
    --------
    dict
        GARCH model parameters
    """
    try:
        from arch import arch_model

        if isinstance(returns, pd.Series):
            returns = returns.values

        # Fit GARCH model
        model = arch_model(returns, vol='GARCH', p=p, q=q)
        result = model.fit(disp='off')

        # Extract parameters
        params = {
            'omega': result.params['omega'],
            'alpha': result.params[f'alpha[1]'] if p > 0 else 0,
            'beta': result.params[f'beta[1]'] if q > 0 else 0,
            'persistence': result.params[f'alpha[1]'] + result.params[f'beta[1]'] if p > 0 and q > 0 else 0,
            'unconditional_variance': result.unconditional_variance,
            'aic': result.aic,
            'bic': result.bic,
            'log_likelihood': result.loglikelihood
        }

        return params

    except ImportError:
        raise ImportError("arch package is required for GARCH modeling")


# Correlation analysis
def calculate_correlation_matrix(df, method='pearson'):
    """
    Calculate correlation matrix for DataFrame columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time series
    method : str
        Correlation method ('pearson', 'spearman', or 'kendall')

    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    return df.corr(method=method)


def calculate_rolling_correlation(series1, series2, window=30, method='pearson'):
    """
    Calculate rolling correlation between two series.

    Parameters:
    -----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series
    window : int
        Rolling window size
    method : str
        Correlation method ('pearson', 'spearman', or 'kendall')

    Returns:
    --------
    pd.Series
        Rolling correlation
    """
    if not isinstance(series1, pd.Series):
        series1 = pd.Series(series1)

    if not isinstance(series2, pd.Series):
        series2 = pd.Series(series2)

    # Align series
    combined = pd.DataFrame({'series1': series1, 'series2': series2})

    # Calculate rolling correlation
    if method == 'pearson':
        return combined['series1'].rolling(window=window).corr(combined['series2'])
    else:
        # For non-pearson correlations, we need to apply custom function
        def rolling_corr(x):
            if len(x) < 2:
                return np.nan
            return pd.Series(x.iloc[:, 0]).corr(pd.Series(x.iloc[:, 1]), method=method)

        return combined.rolling(window=window).apply(rolling_corr, raw=False)


# Data visualization helpers
def plot_volatility_comparison(volatilities, labels=None, title="Volatility Comparison", figsize=(12, 6)):
    """
    Plot comparison of multiple volatility series.

    Parameters:
    -----------
    volatilities : list of array-like
        List of volatility series to compare
    labels : list of str, optional
        Labels for each volatility series
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        The plot figure
    """
    import matplotlib.pyplot as plt

    # Create default labels if not provided
    if labels is None:
        labels = [f"Volatility {i + 1}" for i in range(len(volatilities))]

    # Ensure all volatilities are pandas Series with datetime index
    processed_vols = []
    for vol in volatilities:
        if isinstance(vol, pd.Series):
            processed_vols.append(vol)
        else:
            processed_vols.append(pd.Series(vol))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    for vol, label in zip(processed_vols, labels):
        ax.plot(vol.index, vol.values, label=label)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_regime_distribution(regimes, title="Market Regime Distribution", figsize=(10, 6)):
    """
    Plot distribution of market regimes.

    Parameters:
    -----------
    regimes : pd.Series
        Series of regime labels
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        The plot figure
    """
    import matplotlib.pyplot as plt

    # Count regime occurrences
    regime_counts = regimes.value_counts()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    regime_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)

    ax.set_title(title)
    ax.set_ylabel("")

    return fig


def plot_heatmap(data, title="Correlation Heatmap", figsize=(10, 8)):
    """
    Plot heatmap of correlation matrix or other 2D data.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with numeric values
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        The plot figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(data, annot=True, cmap='coolwarm', center=0, ax=ax)

    ax.set_title(title)

    return fig