"""
Module for normalizing and preprocessing data.
Handles data cleaning, transformation, and standardization.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from datetime import datetime, timedelta
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_data_logger

# Set up logger
logger = get_data_logger()


class DataNormalizer:
    """
    Normalizer for time series data.

    This class handles data cleaning, transformation, and standardization
    to prepare data for analysis and modeling.
    """

    def __init__(self, scaler_type='standard'):
        """
        Initialize the data normalizer.

        Parameters:
        -----------
        scaler_type : str
            Type of scaler to use ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scalers = {}

        logger.info(f"DataNormalizer initialized with {scaler_type} scaler")

    def _get_scaler(self, key):
        """
        Get a scaler for a specific key.

        Parameters:
        -----------
        key : str
            Identifier for the scaler

        Returns:
        --------
        sklearn.preprocessing.Scaler
            Scaler instance
        """
        if key not in self.scalers:
            if self.scaler_type == 'standard':
                self.scalers[key] = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scalers[key] = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scalers[key] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        return self.scalers[key]

    def clean_ohlcv_data(self, df, fill_method='ffill'):
        """
        Clean OHLCV data by handling missing values and outliers.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        fill_method : str
            Method to fill missing values ('ffill', 'bfill', 'interpolate')

        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to clean_ohlcv_data")
            return df

        # Make a copy to avoid modifying the original
        df_clean = df.copy()

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]

        if missing_cols:
            logger.warning(f"Missing required columns in OHLCV data: {missing_cols}")
            return df_clean

        # Check for NaN values
        nan_counts = df_clean[required_cols].isna().sum()

        if nan_counts.sum() > 0:
            logger.info(f"Filling {nan_counts.sum()} NaN values in OHLCV data")

            # Fill NaN values
            if fill_method == 'ffill':
                df_clean[required_cols] = df_clean[required_cols].fillna(method='ffill')
                # Fill any remaining NaNs (at the beginning) with backward fill
                df_clean[required_cols] = df_clean[required_cols].fillna(method='bfill')
            elif fill_method == 'bfill':
                df_clean[required_cols] = df_clean[required_cols].fillna(method='bfill')
                # Fill any remaining NaNs (at the end) with forward fill
                df_clean[required_cols] = df_clean[required_cols].fillna(method='ffill')
            elif fill_method == 'interpolate':
                df_clean[required_cols] = df_clean[required_cols].interpolate(method='linear')
                # Fill any remaining NaNs at the edges
                df_clean[required_cols] = df_clean[required_cols].fillna(method='ffill').fillna(method='bfill')
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")

        # Check for outliers in price columns
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            # Calculate rolling median and MAD (Median Absolute Deviation)
            rolling_median = df_clean[col].rolling(window=48, min_periods=1).median()
            rolling_mad = (df_clean[col] - rolling_median).abs().rolling(window=48, min_periods=1).median()

            # Define outliers as values more than 5 MADs from the median
            outliers = (df_clean[col] - rolling_median).abs() > 5 * rolling_mad
            outlier_count = outliers.sum()

            if outlier_count > 0:
                logger.info(f"Replacing {outlier_count} outliers in {col}")

                # Replace outliers with the rolling median
                df_clean.loc[outliers, col] = rolling_median[outliers]

        # Ensure high >= open, close, low and low <= open, close
        # high should be the maximum of open, close, and itself
        df_clean['high'] = df_clean[['high', 'open', 'close']].max(axis=1)

        # low should be the minimum of open, close, and itself
        df_clean['low'] = df_clean[['low', 'open', 'close']].min(axis=1)

        # Ensure volume is non-negative
        df_clean['volume'] = df_clean['volume'].clip(lower=0)

        # Check for zero prices
        zero_prices = (df_clean[price_cols] <= 0).any(axis=1)

        if zero_prices.sum() > 0:
            logger.warning(f"Found {zero_prices.sum()} rows with zero or negative prices")

            # Replace with previous valid prices
            df_clean.loc[zero_prices, price_cols] = df_clean.loc[~zero_prices, price_cols].ffill()

        return df_clean

    def normalize_price_data(self, df, columns=None, key=None):
        """
        Normalize price data using the specified scaler.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        columns : list, optional
            Columns to normalize (if None, uses ['open', 'high', 'low', 'close'])
        key : str, optional
            Identifier for the scaler (if None, uses 'price')

        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized price data
        tuple
            (normalized DataFrame, original DataFrame)
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to normalize_price_data")
            return df, df

        # Make a copy to avoid modifying the original
        df_orig = df.copy()
        df_norm = df.copy()

        # Default columns
        if columns is None:
            columns = ['open', 'high', 'low', 'close']

        # Check for required columns
        missing_cols = [col for col in columns if col not in df_norm.columns]

        if missing_cols:
            logger.warning(f"Missing columns for normalization: {missing_cols}")
            # Only use available columns
            columns = [col for col in columns if col in df_norm.columns]

        if not columns:
            logger.warning("No columns available for normalization")
            return df_norm, df_orig

        # Get scaler
        key = key or 'price'
        scaler = self._get_scaler(key)

        # Fit and transform
        try:
            values = df_norm[columns].values
            normalized_values = scaler.fit_transform(values)

            # Update DataFrame
            for i, col in enumerate(columns):
                df_norm[f"{col}_norm"] = normalized_values[:, i]

            logger.info(f"Normalized {len(columns)} price columns")
            return df_norm, df_orig

        except Exception as e:
            logger.error(f"Error normalizing price data: {e}")
            return df_norm, df_orig

    def normalize_volume_data(self, df, column='volume', key=None):
        """
        Normalize volume data using the specified scaler.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volume data
        column : str
            Volume column name
        key : str, optional
            Identifier for the scaler (if None, uses 'volume')

        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized volume data
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to normalize_volume_data")
            return df

        # Make a copy to avoid modifying the original
        df_norm = df.copy()

        # Check for required column
        if column not in df_norm.columns:
            logger.warning(f"Volume column '{column}' not found in DataFrame")
            return df_norm

        # Get scaler
        key = key or 'volume'
        scaler = self._get_scaler(key)

        # Apply log transformation first (volume often has a skewed distribution)
        df_norm[f"{column}_log"] = np.log1p(df_norm[column])

        # Fit and transform
        try:
            values = df_norm[[f"{column}_log"]].values
            normalized_values = scaler.fit_transform(values)

            # Update DataFrame
            df_norm[f"{column}_norm"] = normalized_values

            logger.info(f"Normalized volume data")
            return df_norm

        except Exception as e:
            logger.error(f"Error normalizing volume data: {e}")
            return df_norm

    def calculate_returns(self, df, price_col='close', method='log'):
        """
        Calculate returns from price data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data
        method : str
            Method to calculate returns ('log' or 'pct')

        Returns:
        --------
        pd.DataFrame
            DataFrame with returns
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_returns")
            return df

        # Make a copy to avoid modifying the original
        df_returns = df.copy()

        # Check for required column
        if price_col not in df_returns.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_returns

        # Calculate returns
        try:
            if method == 'log':
                df_returns[f"{price_col}_log_return"] = np.log(df_returns[price_col] / df_returns[price_col].shift(1))
            elif method == 'pct':
                df_returns[f"{price_col}_pct_return"] = df_returns[price_col].pct_change()
            else:
                raise ValueError(f"Unknown return calculation method: {method}")

            # Drop NaN values from the first row
            if len(df_returns) > 1:
                df_returns.iloc[0, df_returns.columns.get_loc(
                    f"{price_col}_log_return" if method == 'log' else f"{price_col}_pct_return")] = 0

            logger.info(f"Calculated {method} returns for {price_col}")
            return df_returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return df_returns

    def calculate_technical_indicators(self, df, price_col='close', volume_col='volume'):
        """
        Calculate technical indicators from OHLCV data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        price_col : str
            Column name for price data
        volume_col : str
            Column name for volume data

        Returns:
        --------
        pd.DataFrame
            DataFrame with technical indicators
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_technical_indicators")
            return df

        # Make a copy to avoid modifying the original
        df_indicators = df.copy()

        # Check for required columns
        missing_cols = []

        if price_col not in df_indicators.columns:
            missing_cols.append(price_col)

        if volume_col not in df_indicators.columns:
            missing_cols.append(volume_col)

        if missing_cols:
            logger.warning(f"Missing required columns for indicators: {missing_cols}")
            return df_indicators

        try:
            # 1. Moving Averages
            # Simple Moving Average (SMA)
            df_indicators[f'{price_col}_sma_5'] = df_indicators[price_col].rolling(window=5).mean()
            df_indicators[f'{price_col}_sma_20'] = df_indicators[price_col].rolling(window=20).mean()
            df_indicators[f'{price_col}_sma_50'] = df_indicators[price_col].rolling(window=50).mean()

            # Exponential Moving Average (EMA)
            df_indicators[f'{price_col}_ema_5'] = df_indicators[price_col].ewm(span=5, adjust=False).mean()
            df_indicators[f'{price_col}_ema_20'] = df_indicators[price_col].ewm(span=20, adjust=False).mean()
            df_indicators[f'{price_col}_ema_50'] = df_indicators[price_col].ewm(span=50, adjust=False).mean()

            # 2. Volatility Indicators
            # Bollinger Bands
            df_indicators[f'{price_col}_sma_20'] = df_indicators[price_col].rolling(window=20).mean()
            df_indicators[f'{price_col}_std_20'] = df_indicators[price_col].rolling(window=20).std()
            df_indicators[f'{price_col}_bollinger_upper'] = df_indicators[f'{price_col}_sma_20'] + 2 * df_indicators[
                f'{price_col}_std_20']
            df_indicators[f'{price_col}_bollinger_lower'] = df_indicators[f'{price_col}_sma_20'] - 2 * df_indicators[
                f'{price_col}_std_20']

            # Average True Range (ATR)
            high = df_indicators['high'] if 'high' in df_indicators.columns else df_indicators[price_col]
            low = df_indicators['low'] if 'low' in df_indicators.columns else df_indicators[price_col]
            close_prev = df_indicators[price_col].shift(1)

            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_indicators['atr_14'] = tr.rolling(window=14).mean()

            # 3. Momentum Indicators
            # Relative Strength Index (RSI)
            delta = df_indicators[price_col].diff()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            df_indicators['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD (Moving Average Convergence Divergence)
            ema_12 = df_indicators[price_col].ewm(span=12, adjust=False).mean()
            ema_26 = df_indicators[price_col].ewm(span=26, adjust=False).mean()
            df_indicators['macd'] = ema_12 - ema_26
            df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
            df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']

            # 4. Volume Indicators
            # On-Balance Volume (OBV)
            obv = (np.sign(df_indicators[price_col].diff()) * df_indicators[volume_col]).fillna(0).cumsum()
            df_indicators['obv'] = obv

            # Volume Moving Average
            df_indicators[f'{volume_col}_sma_20'] = df_indicators[volume_col].rolling(window=20).mean()

            # 5. Trend Indicators
            # Average Directional Index (ADX)
            # This is a simplified version
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
            minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)

            tr_14 = tr.rolling(window=14).sum()
            plus_di_14 = 100 * (plus_dm.rolling(window=14).sum() / tr_14)
            minus_di_14 = 100 * (minus_dm.rolling(window=14).sum() / tr_14)

            dx = 100 * ((plus_di_14 - minus_di_14).abs() / (plus_di_14 + minus_di_14).abs())
            df_indicators['adx_14'] = dx.rolling(window=14).mean()

            logger.info(f"Calculated technical indicators")
            return df_indicators

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df_indicators

    def normalize_dataframe(self, df, columns=None, key=None):
        """
        Normalize specified columns in a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to normalize
        columns : list, optional
            Columns to normalize (if None, uses all numeric columns)
        key : str, optional
            Identifier for the scaler (if None, uses 'dataframe')

        Returns:
        --------
        pd.DataFrame
            Normalized DataFrame
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to normalize_dataframe")
            return df

        # Make a copy to avoid modifying the original
        df_norm = df.copy()

        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()

        # Check for required columns
        missing_cols = [col for col in columns if col not in df_norm.columns]

        if missing_cols:
            logger.warning(f"Missing columns for normalization: {missing_cols}")
            # Only use available columns
            columns = [col for col in columns if col in df_norm.columns]

        if not columns:
            logger.warning("No columns available for normalization")
            return df_norm

        # Get scaler
        key = key or 'dataframe'
        scaler = self._get_scaler(key)

        # Fit and transform
        try:
            values = df_norm[columns].values
            normalized_values = scaler.fit_transform(values)

            # Create new normalized columns
            for i, col in enumerate(columns):
                df_norm[f"{col}_norm"] = normalized_values[:, i]

            logger.info(f"Normalized {len(columns)} columns")
            return df_norm

        except Exception as e:
            logger.error(f"Error normalizing dataframe: {e}")
            return df_norm

    def prepare_features(self, df, price_col='close', volume_col='volume', target_col=None, window=None):
        """
        Prepare features for machine learning.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with raw data
        price_col : str
            Column name for price data
        volume_col : str
            Column name for volume data
        target_col : str, optional
            Column name for target variable (if None, uses price returns)
        window : int, optional
            Window size for rolling features (if None, uses default sizes)

        Returns:
        --------
        pd.DataFrame
            DataFrame with prepared features
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to prepare_features")
            return df

        # Clean data
        df_clean = self.clean_ohlcv_data(df)

        # Calculate returns
        df_returns = self.calculate_returns(df_clean, price_col=price_col)

        # Calculate technical indicators
        df_indicators = self.calculate_technical_indicators(df_returns, price_col=price_col, volume_col=volume_col)

        # Define windows for rolling features
        if window is None:
            windows = [5, 10, 20, 50]
        else:
            windows = [window]

        # Calculate rolling features
        for w in windows:
            # Rolling mean of returns
            return_col = f"{price_col}_log_return"
            if return_col in df_indicators.columns:
                df_indicators[f"{return_col}_mean_{w}"] = df_indicators[return_col].rolling(window=w).mean()
                df_indicators[f"{return_col}_std_{w}"] = df_indicators[return_col].rolling(window=w).std()

            # Rolling z-score of price
            df_indicators[f"{price_col}_zscore_{w}"] = (
                    (df_indicators[price_col] - df_indicators[price_col].rolling(window=w).mean()) /
                    df_indicators[price_col].rolling(window=w).std()
            )

            # Rolling ratio to moving average
            df_indicators[f"{price_col}_ratio_sma_{w}"] = df_indicators[price_col] / df_indicators[price_col].rolling(
                window=w).mean()

            # Rolling volume features
            df_indicators[f"{volume_col}_mean_{w}"] = df_indicators[volume_col].rolling(window=w).mean()
            df_indicators[f"{volume_col}_std_{w}"] = df_indicators[volume_col].rolling(window=w).std()
            df_indicators[f"{volume_col}_zscore_{w}"] = (
                    (df_indicators[volume_col] - df_indicators[volume_col].rolling(window=w).mean()) /
                    df_indicators[volume_col].rolling(window=w).std()
            )

        # Define target variable
        if target_col is None:
            # Default: next period return
            return_col = f"{price_col}_log_return"
            if return_col in df_indicators.columns:
                df_indicators['target'] = df_indicators[return_col].shift(-1)
        else:
            if target_col in df_indicators.columns:
                df_indicators['target'] = df_indicators[target_col]
            else:
                logger.warning(f"Target column '{target_col}' not found in DataFrame")

        # Handle NaN values
        df_indicators = df_indicators.fillna(method='ffill').fillna(method='bfill')

        # Drop rows with remaining NaN values
        df_features = df_indicators.dropna()

        logger.info(f"Prepared features with {df_features.shape[1]} columns")
        return df_features


# Factory function to get a data normalizer
def get_data_normalizer(scaler_type='standard'):
    """
    Get a configured data normalizer.

    Parameters:
    -----------
    scaler_type : str
        Type of scaler to use ('standard', 'minmax', 'robust')

    Returns:
    --------
    DataNormalizer
        Configured normalizer instance
    """
    return DataNormalizer(scaler_type=scaler_type)